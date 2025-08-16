# BTCW OpenCL GPU Miner - AMD Compatible

Acest este un port OpenCL al minerului BTCW original CUDA, optimizat pentru GPU-urile AMD.

## Cerințe de sistem

### AMD GPU:
- Driver AMD Radeon Software Adrenalin
- OpenCL 2.0 sau mai nou
- ROCm (opțional, pentru performanță optimă)

### Dependințe:
- CMake 3.16+
- OpenCL headers și libraries
- ncurses development libraries
- GCC/Clang cu suport C++17

## Instalare dependințe

### Ubuntu/Debian:
```bash
sudo apt update
sudo apt install cmake build-essential libncurses5-dev libncursesw5-dev
sudo apt install opencl-headers ocl-icd-opencl-dev
sudo apt install mesa-opencl-icd  # Pentru AMD GPU
```

### Arch Linux:
```bash
sudo pacman -S cmake base-devel ncurses
sudo pacman -S opencl-headers opencl-icd-loader
sudo pacman -S opencl-mesa  # Pentru AMD GPU
```

### CentOS/RHEL/Fedora:
```bash
sudo dnf install cmake gcc-c++ ncurses-devel
sudo dnf install opencl-headers ocl-icd-devel
sudo dnf install mesa-libOpenCL  # Pentru AMD GPU
```

## Compilare

```bash
# Clonează repository-ul
git clone <repository-url>
cd gpu_miner_btcw_opencl

# Fă scriptul de build executabil
chmod +x build.sh

# Compilează
./build.sh
```

Sau manual:
```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Utilizare

1. **Pornește nodul BTCW** cu wallet-ul tău:
   ```bash
   # Asigură-te că ai cel puțin 1 UTXO în wallet
   ./btcw-node --mine
   ```

2. **Pornește minerul OpenCL**:
   ```bash
   # Pentru primul GPU
   ./build/gpu_miner_opencl 1
   
   # Pentru al doilea GPU
   ./build/gpu_miner_opencl 2
   
   # etc.
   ```

## Caracteristici

- **Compatibilitate AMD**: Complet portat de la CUDA la OpenCL
- **Performanță optimizată**: Kernel-uri optimizate pentru arhitectura AMD
- **Interfață ncurses**: Afișare în timp real a hashrate-ului și statusului
- **Shared memory**: Comunicare cu nodul BTCW prin memoria partajată
- **Multi-GPU**: Suport pentru multiple GPU-uri AMD

## Optimizări pentru AMD

- Work group size optimizat pentru GPU-urile AMD (256)
- Utilizare eficientă a memoriei locale
- Algoritmi de hashing optimizați pentru arhitectura GCN/RDNA
- Gestionare îmbunătățită a thread-urilor

## Depanare

### GPU-ul nu este detectat:
```bash
# Verifică dacă OpenCL detectează GPU-ul
clinfo

# Verifică driver-ele AMD
lspci | grep -i amd
```

### Erori de compilare:
- Asigură-te că ai toate dependințele instalate
- Verifică că ai OpenCL headers și libraries
- Pentru AMD GPU, instalează mesa-opencl-icd sau ROCm

### Performanță scăzută:
- Verifică că folosești driver-ele AMD oficiale
- Încearcă să ajustezi `globalWorkSize` și `localWorkSize` în cod
- Pentru GPU-uri mai noi, consideră ROCm în loc de mesa-opencl

## Diferențe față de versiunea CUDA

- **Kernel language**: OpenCL C în loc de CUDA C
- **Memory management**: clCreateBuffer în loc de cudaMalloc
- **Kernel execution**: clEnqueueNDRangeKernel în loc de <<<>>>
- **Platform detection**: Detectare automată AMD GPU
- **Work organization**: Global/local work items în loc de grid/block

## Contribuții

Contribuțiile sunt binevenite! Te rog să:
1. Faci fork la repository
2. Creezi un branch pentru feature-ul tău
3. Faci commit cu modificările
4. Deschizi un Pull Request

## Licență

Același ca proiectul original BTCW.

## Suport

Pentru probleme specifice OpenCL/AMD, deschide un issue cu:
- Informații despre GPU (clinfo output)
- Versiunea driver-ului AMD
- Distribuția Linux folosită
- Log-urile de eroare complete
