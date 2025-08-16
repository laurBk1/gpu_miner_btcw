#include <iostream>
#include <fcntl.h>
#include <sys/mman.h>
#include <semaphore.h>
#include <unistd.h>
#include <vector>
#include <random>
#include <cstring>
#include <ncurses.h>
#include <fstream>
#include <chrono>
#include <thread>

#include "opencl_utils.h"
#include "sha256.h"

#define SHM_NAME "/shared_mem"

const int CTX_SIZE_BYTES = 8*20; // 160
const int KEY_SIZE_BYTES = 32;
const int HASH_NO_SIG_SIZE_BYTES = 32;
const int TOTAL_BYTES_SEND = CTX_SIZE_BYTES + KEY_SIZE_BYTES + HASH_NO_SIG_SIZE_BYTES;
const int NONCE_SIZE_BYTES = 8;

struct SharedData {
    volatile uint64_t nonce;
    volatile uint8_t data[TOTAL_BYTES_SEND];
};

std::string loadKernelSource(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open kernel file: " << filename << std::endl;
        return "";
    }
    
    std::string source((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
    return source;
}

volatile uint64_t nonce4hashrate = 0;
volatile uint64_t nonce4hashrate_prev = 0;

int main(int argc, char* argv[]) {
    int gpu_num = 0; // default
    if (argc == 2) {
        gpu_num = atoi(argv[1]);
    }

    std::cout << "BTCW OpenCL GPU Miner v1.0.0 - AMD Compatible" << std::endl;
    std::cout << "GPU Number: " << gpu_num << std::endl;

    // Initialize OpenCL
    OpenCLManager ocl;
    if (!ocl.initialize()) {
        std::cerr << "Failed to initialize OpenCL" << std::endl;
        return 1;
    }

    ocl.printDeviceInfo();

    // Load and compile kernel
    std::string kernelSource = loadKernelSource("../kernels/mining_kernel.cl");
    if (kernelSource.empty()) {
        std::cerr << "Failed to load kernel source" << std::endl;
        return 1;
    }

    if (!ocl.createKernel(kernelSource, "btcw_miner")) {
        std::cerr << "Failed to create kernel" << std::endl;
        return 1;
    }

    // Allocate host memory
    uint8_t h_gpu_num = static_cast<uint8_t>(gpu_num);
    uint8_t h_ctx_data[CTX_SIZE_BYTES];
    uint8_t h_key_data[KEY_SIZE_BYTES];
    uint8_t h_hash_no_sig_data[HASH_NO_SIG_SIZE_BYTES];
    uint64_t h_nonce_data = 0;
    uint64_t h_nonce4hashrate_data = 0;

    // Create OpenCL buffers
    cl_mem d_gpu_num = ocl.createBuffer(1, CL_MEM_READ_ONLY);
    cl_mem d_ctx_data = ocl.createBuffer(CTX_SIZE_BYTES, CL_MEM_READ_ONLY);
    cl_mem d_key_data = ocl.createBuffer(KEY_SIZE_BYTES, CL_MEM_READ_ONLY);
    cl_mem d_hash_no_sig_data = ocl.createBuffer(HASH_NO_SIG_SIZE_BYTES, CL_MEM_READ_ONLY);
    cl_mem d_nonce_data = ocl.createBuffer(NONCE_SIZE_BYTES, CL_MEM_READ_WRITE);
    cl_mem d_nonce4hashrate_data = ocl.createBuffer(NONCE_SIZE_BYTES, CL_MEM_READ_WRITE);

    if (!d_gpu_num || !d_ctx_data || !d_key_data || !d_hash_no_sig_data || 
        !d_nonce_data || !d_nonce4hashrate_data) {
        std::cerr << "Failed to create OpenCL buffers" << std::endl;
        return 1;
    }

    // Write initial data
    ocl.writeBuffer(d_gpu_num, 1, &h_gpu_num);
    ocl.writeBuffer(d_nonce_data, NONCE_SIZE_BYTES, &h_nonce_data);
    ocl.writeBuffer(d_nonce4hashrate_data, NONCE_SIZE_BYTES, &h_nonce4hashrate_data);

    // Set kernel arguments
    std::vector<void*> args = {&d_gpu_num, &d_key_data, &d_ctx_data, 
                               &d_hash_no_sig_data, &d_nonce_data, &d_nonce4hashrate_data};
    std::vector<size_t> argSizes = {sizeof(cl_mem), sizeof(cl_mem), sizeof(cl_mem),
                                    sizeof(cl_mem), sizeof(cl_mem), sizeof(cl_mem)};

    if (!ocl.setKernelArgs(args, argSizes)) {
        std::cerr << "Failed to set kernel arguments" << std::endl;
        return 1;
    }

    // Open shared memory
    int shm_fd = shm_open(SHM_NAME, O_RDWR, 0666);
    SharedData* shared_data = nullptr;
    
    if (shm_fd != -1) {
        shared_data = (SharedData*) mmap(NULL, sizeof(SharedData), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
        if (shared_data == MAP_FAILED) {
            shared_data = nullptr;
        }
    }

    // Initialize ncurses
    initscr();
    noecho();
    curs_set(FALSE);

    int prev_y, prev_x;
    int curr_y = 0, curr_x = 0;
    getmaxyx(stdscr, prev_y, prev_x);
    mvprintw(0, 0, "Bitcoin-PoW OpenCL Miner v1.0.0\n");

    volatile uint64_t nonce_prev = 1234;
    nonce4hashrate_prev = 12345;
    static uint64_t hash_no_sig = 0;
    uint32_t throttle = 0;

    // Mining parameters
    const size_t globalWorkSize = 32768;  // Total work items
    const size_t localWorkSize = 256;     // Work group size

    while (true) {
        int changeCount = 0;
        const int durationSeconds = 2;
        auto startTime = std::chrono::steady_clock::now();

        while (std::chrono::steady_clock::now() - startTime < std::chrono::seconds(durationSeconds)) {
            
            if ((throttle % 0x3) == 0 && shared_data) {
                // Update data from BTCW node
                memcpy(&h_key_data[0], const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[0])), 32);

                // Copy context data
                for (int i = 0; i < 20; i++) {
                    memcpy(&h_ctx_data[i*8], const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[32 + i*8])), 8);
                }

                // Copy hash_no_sig data
                memcpy(&h_hash_no_sig_data[0], const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[192])), 32);

                // Write updated data to GPU
                ocl.writeBuffer(d_ctx_data, CTX_SIZE_BYTES, h_ctx_data);
                ocl.writeBuffer(d_key_data, KEY_SIZE_BYTES, h_key_data);
                ocl.writeBuffer(d_hash_no_sig_data, HASH_NO_SIG_SIZE_BYTES, h_hash_no_sig_data);
            }

            throttle++;

            // Execute kernel
            if (!ocl.executeKernel(globalWorkSize, localWorkSize)) {
                std::cerr << "Failed to execute kernel" << std::endl;
                break;
            }

            // Read results
            if (shared_data) {
                ocl.readBuffer(d_nonce_data, NONCE_SIZE_BYTES, const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->nonce)));
            }
            ocl.readBuffer(d_nonce4hashrate_data, NONCE_SIZE_BYTES, const_cast<void*>(reinterpret_cast<const volatile void*>(&nonce4hashrate)));

            // Update display
            getmaxyx(stdscr, curr_y, curr_x);

            if (curr_y != prev_y || curr_x != prev_x) {
                clear();
                prev_y = curr_y;
                prev_x = curr_x;
                mvprintw(0, 0, "Bitcoin-PoW OpenCL Miner v1.0.0\n");
            }

            if (shared_data && nonce_prev != shared_data->nonce) {
                nonce_prev = shared_data->nonce;
                mvprintw(2, 0, "Hash found - NONCE: %016lx\n", nonce_prev);
            }

            if (shared_data) {
                memcpy(&hash_no_sig, const_cast<void*>(reinterpret_cast<const volatile void*>(&shared_data->data[192])), 8);
                mvprintw(4, 0, "Hash no sig low64: %016lx\n", hash_no_sig);

                if (hash_no_sig == 0) {
                    mvprintw(6, 0, "!!! NOT CONNECTED TO BTCW NODE WALLET !!!  ---> Make sure your wallet has at least 1 utxo.\n");
                    mvprintw(7, 0, "!!! NOT CONNECTED TO BTCW NODE WALLET !!!  ---> Make sure your wallet has at least 1 utxo.\n");
                    mvprintw(8, 0, "!!! NOT CONNECTED TO BTCW NODE WALLET !!!  ---> Make sure your wallet has at least 1 utxo.\n");

                    // Try to reconnect
                    if (shm_fd == -1) {
                        shm_fd = shm_open(SHM_NAME, O_RDWR, 0666);
                        if (shm_fd != -1) {
                            shared_data = (SharedData*) mmap(NULL, sizeof(SharedData), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
                            if (shared_data == MAP_FAILED) {
                                shared_data = nullptr;
                            }
                        }
                    }
                    usleep(1000000);
                } else {
                    mvprintw(6, 0, "CONNECTED TO BTCW NODE WALLET\n");
                    mvprintw(7, 0, "\n");
                    mvprintw(8, 0, "\n");
                }
            } else {
                mvprintw(6, 0, "!!! NOT CONNECTED TO BTCW NODE WALLET !!!  ---> Make sure your wallet has at least 1 utxo.\n");
                mvprintw(7, 0, "!!! NOT CONNECTED TO BTCW NODE WALLET !!!  ---> Make sure your wallet has at least 1 utxo.\n");
                mvprintw(8, 0, "!!! NOT CONNECTED TO BTCW NODE WALLET !!!  ---> Make sure your wallet has at least 1 utxo.\n");
            }

            if (nonce4hashrate != nonce4hashrate_prev) {
                changeCount++;
                nonce4hashrate_prev = nonce4hashrate;
            }

            usleep(50);
        }

        double rate = static_cast<double>(changeCount) / durationSeconds;
        rate *= 65536;

        if (hash_no_sig == 0) {
            rate = 0;
        }

        // Display status
        mvprintw(curr_y - 5, 0, "=======================================================\n");
        mvprintw(curr_y - 4, 0, "Device: OpenCL AMD GPU\n");
        mvprintw(curr_y - 3, 0, "-------------------------------------------------------\n");
        mvprintw(curr_y - 2, 0, "Hashrate: %lf H/s\n", rate);
        mvprintw(curr_y - 1, 0, "=======================================================\n");
        refresh();
    }

    // Cleanup
    ocl.releaseBuffer(d_gpu_num);
    ocl.releaseBuffer(d_ctx_data);
    ocl.releaseBuffer(d_key_data);
    ocl.releaseBuffer(d_hash_no_sig_data);
    ocl.releaseBuffer(d_nonce_data);
    ocl.releaseBuffer(d_nonce4hashrate_data);

    if (shared_data) {
        munmap(shared_data, sizeof(SharedData));
    }
    if (shm_fd != -1) {
        close(shm_fd);
    }

    endwin();
    return 0;
}