// Bitcoin-PoW OpenCL Mining Kernel
// Ported from CUDA to OpenCL for AMD GPU compatibility

#define ROTRIGHT(a,b) (((a) >> (b)) | ((a) << (32-(b))))
#define CH(x,y,z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTRIGHT(x,2) ^ ROTRIGHT(x,13) ^ ROTRIGHT(x,22))
#define EP1(x) (ROTRIGHT(x,6) ^ ROTRIGHT(x,11) ^ ROTRIGHT(x,25))
#define SIG0(x) (ROTRIGHT(x,7) ^ ROTRIGHT(x,18) ^ ((x) >> 3))
#define SIG1(x) (ROTRIGHT(x,17) ^ ROTRIGHT(x,19) ^ ((x) >> 10))

__constant uint k[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

void sha256_transform(__private uint* state, __private const uchar* data) {
    uint a, b, c, d, e, f, g, h, i, j, t1, t2, m[64];

    for (i = 0, j = 0; i < 16; ++i, j += 4)
        m[i] = (data[j] << 24) | (data[j + 1] << 16) | (data[j + 2] << 8) | (data[j + 3]);
    for ( ; i < 64; ++i)
        m[i] = SIG1(m[i - 2]) + m[i - 7] + SIG0(m[i - 15]) + m[i - 16];

    a = state[0]; b = state[1]; c = state[2]; d = state[3];
    e = state[4]; f = state[5]; g = state[6]; h = state[7];

    for (i = 0; i < 64; ++i) {
        t1 = h + EP1(e) + CH(e,f,g) + k[i] + m[i];
        t2 = EP0(a) + MAJ(a,b,c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

void sha256_hash(__private const uchar* data, uint len, __private uchar* hash) {
    uint state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    
    uchar buffer[64];
    uint buflen = 0;
    ulong bitlen = 0;
    
    // Process input data
    for (uint i = 0; i < len; i++) {
        buffer[buflen++] = data[i];
        if (buflen == 64) {
            sha256_transform(state, buffer);
            bitlen += 512;
            buflen = 0;
        }
    }
    
    // Padding
    buffer[buflen++] = 0x80;
    if (buflen > 56) {
        while (buflen < 64) buffer[buflen++] = 0;
        sha256_transform(state, buffer);
        buflen = 0;
    }
    while (buflen < 56) buffer[buflen++] = 0;
    
    // Append length
    bitlen += (len % 64) * 8;
    for (int i = 7; i >= 0; i--) {
        buffer[56 + i] = bitlen & 0xff;
        bitlen >>= 8;
    }
    
    sha256_transform(state, buffer);
    
    // Output hash
    for (int i = 0; i < 8; i++) {
        hash[i*4]     = (state[i] >> 24) & 0xff;
        hash[i*4 + 1] = (state[i] >> 16) & 0xff;
        hash[i*4 + 2] = (state[i] >> 8) & 0xff;
        hash[i*4 + 3] = state[i] & 0xff;
    }
}

bool check_target(__private const uchar* hash, __private const uchar* target) {
    for (int i = 31; i >= 0; i--) {
        if (hash[i] > target[i]) return false;
        if (hash[i] < target[i]) return true;
    }
    return true;
}

__kernel void btcw_miner(
    __global uchar* gpu_num,
    __global uchar* key_data,
    __global uchar* ctx_data,
    __global uchar* hash_no_sig_data,
    __global ulong* nonce_result,
    __global ulong* nonce_counter
) {
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    
    // Each work item gets a unique nonce range
    ulong base_nonce = ((ulong)gid) << 16;
    ulong local_counter = 0;
    
    // Local copies of data
    uchar local_key[32];
    uchar local_ctx[160];
    uchar local_hash_no_sig[32];
    uchar local_gpu_num = gpu_num[0];
    
    // Copy global data to local
    for (int i = 0; i < 32; i++) {
        local_key[i] = key_data[i];
        local_hash_no_sig[i] = hash_no_sig_data[i];
    }
    for (int i = 0; i < 160; i++) {
        local_ctx[i] = ctx_data[i];
    }
    
    // Mining loop - each work item tries 65536 nonces
    for (uint nonce_offset = 0; nonce_offset < 65536; nonce_offset++) {
        ulong current_nonce = base_nonce + nonce_offset + (local_gpu_num * 0x1000000UL);
        
        // Prepare mining data
        uchar mining_data[128];
        
        // Copy key data
        for (int i = 0; i < 32; i++) {
            mining_data[i] = local_key[i];
        }
        
        // Add nonce to mining data
        mining_data[32] = (current_nonce >> 56) & 0xff;
        mining_data[33] = (current_nonce >> 48) & 0xff;
        mining_data[34] = (current_nonce >> 40) & 0xff;
        mining_data[35] = (current_nonce >> 32) & 0xff;
        mining_data[36] = (current_nonce >> 24) & 0xff;
        mining_data[37] = (current_nonce >> 16) & 0xff;
        mining_data[38] = (current_nonce >> 8) & 0xff;
        mining_data[39] = current_nonce & 0xff;
        
        // Add context data
        for (int i = 0; i < 32; i++) {
            mining_data[40 + i] = local_hash_no_sig[i];
        }
        
        // Add some context data
        for (int i = 0; i < 56; i++) {
            mining_data[72 + i] = local_ctx[i];
        }
        
        // Compute hash
        uchar hash_result[32];
        sha256_hash(mining_data, 128, hash_result);
        
        // Double SHA256 for Bitcoin-style hashing
        uchar final_hash[32];
        sha256_hash(hash_result, 32, final_hash);
        
        local_counter++;
        
        // Check if we found a valid hash (simplified difficulty check)
        // Looking for hash with leading zeros
        if (final_hash[0] == 0 && final_hash[1] == 0 && final_hash[2] < 16) {
            // Found a potential solution
            atomic_xchg(nonce_result, current_nonce);
        }
        
        // Update counter every 1024 iterations
        if ((nonce_offset & 0x3ff) == 0) {
            atomic_add(nonce_counter, 1024);
        }
    }
    
    // Final counter update
    atomic_add(nonce_counter, local_counter & 0x3ff);
}