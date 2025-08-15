#include "opencl_utils.h"
#include <iostream>
#include <fstream>
#include <sstream>

OpenCLManager::OpenCLManager() : initialized(false), platform(nullptr), device(nullptr), 
                                 context(nullptr), queue(nullptr), program(nullptr), kernel(nullptr) {
}

OpenCLManager::~OpenCLManager() {
    if (kernel) clReleaseKernel(kernel);
    if (program) clReleaseProgram(program);
    if (queue) clReleaseCommandQueue(queue);
    if (context) clReleaseContext(context);
}

bool OpenCLManager::initialize() {
    cl_int err;
    
    // Get platform
    cl_uint numPlatforms;
    err = clGetPlatformIDs(0, nullptr, &numPlatforms);
    if (err != CL_SUCCESS || numPlatforms == 0) {
        std::cerr << "No OpenCL platforms found" << std::endl;
        return false;
    }
    
    std::vector<cl_platform_id> platforms(numPlatforms);
    err = clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to get platform IDs" << std::endl;
        return false;
    }
    
    // Try to find AMD platform first, then any GPU
    platform = platforms[0]; // Default to first platform
    for (auto p : platforms) {
        char vendor[256];
        clGetPlatformInfo(p, CL_PLATFORM_VENDOR, sizeof(vendor), vendor, nullptr);
        if (strstr(vendor, "Advanced Micro Devices") || strstr(vendor, "AMD")) {
            platform = p;
            break;
        }
    }
    
    // Get device
    cl_uint numDevices;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
    if (err != CL_SUCCESS || numDevices == 0) {
        // Try CPU if no GPU found
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, nullptr, &numDevices);
        if (err != CL_SUCCESS || numDevices == 0) {
            std::cerr << "No OpenCL devices found" << std::endl;
            return false;
        }
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, nullptr);
    } else {
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    }
    
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to get device ID" << std::endl;
        return false;
    }
    
    // Create context
    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create context: " << getErrorString(err) << std::endl;
        return false;
    }
    
    // Create command queue
    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create command queue: " << getErrorString(err) << std::endl;
        return false;
    }
    
    initialized = true;
    return true;
}

bool OpenCLManager::createKernel(const std::string& kernelSource, const std::string& kernelName) {
    if (!initialized) return false;
    
    cl_int err;
    const char* source = kernelSource.c_str();
    size_t sourceSize = kernelSource.length();
    
    program = clCreateProgramWithSource(context, 1, &source, &sourceSize, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create program: " << getErrorString(err) << std::endl;
        return false;
    }
    
    err = clBuildProgram(program, 1, &device, "-cl-std=CL2.0", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to build program: " << getErrorString(err) << std::endl;
        
        // Get build log
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::vector<char> log(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
        std::cerr << "Build log: " << log.data() << std::endl;
        return false;
    }
    
    kernel = clCreateKernel(program, kernelName.c_str(), &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create kernel: " << getErrorString(err) << std::endl;
        return false;
    }
    
    return true;
}

bool OpenCLManager::setKernelArgs(const std::vector<void*>& args, const std::vector<size_t>& argSizes) {
    if (!kernel || args.size() != argSizes.size()) return false;
    
    for (size_t i = 0; i < args.size(); i++) {
        cl_int err = clSetKernelArg(kernel, i, argSizes[i], args[i]);
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to set kernel arg " << i << ": " << getErrorString(err) << std::endl;
            return false;
        }
    }
    return true;
}

bool OpenCLManager::executeKernel(size_t globalWorkSize, size_t localWorkSize) {
    if (!kernel) return false;
    
    cl_int err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize, &localWorkSize, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to execute kernel: " << getErrorString(err) << std::endl;
        return false;
    }
    
    return true;
}

cl_mem OpenCLManager::createBuffer(size_t size, cl_mem_flags flags, void* hostPtr) {
    if (!initialized) return nullptr;
    
    cl_int err;
    cl_mem buffer = clCreateBuffer(context, flags, size, hostPtr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create buffer: " << getErrorString(err) << std::endl;
        return nullptr;
    }
    return buffer;
}

bool OpenCLManager::writeBuffer(cl_mem buffer, size_t size, void* data) {
    if (!buffer || !queue) return false;
    
    cl_int err = clEnqueueWriteBuffer(queue, buffer, CL_TRUE, 0, size, data, 0, nullptr, nullptr);
    return err == CL_SUCCESS;
}

bool OpenCLManager::readBuffer(cl_mem buffer, size_t size, void* data) {
    if (!buffer || !queue) return false;
    
    cl_int err = clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, size, data, 0, nullptr, nullptr);
    return err == CL_SUCCESS;
}

void OpenCLManager::releaseBuffer(cl_mem buffer) {
    if (buffer) clReleaseMemObject(buffer);
}

void OpenCLManager::printDeviceInfo() {
    if (!device) return;
    
    char name[256], vendor[256];
    cl_ulong globalMem, localMem;
    cl_uint computeUnits, maxWorkGroupSize;
    
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, nullptr);
    clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(vendor), vendor, nullptr);
    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(globalMem), &globalMem, nullptr);
    clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(localMem), &localMem, nullptr);
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(computeUnits), &computeUnits, nullptr);
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkGroupSize), &maxWorkGroupSize, nullptr);
    
    std::cout << "Device: " << name << std::endl;
    std::cout << "Vendor: " << vendor << std::endl;
    std::cout << "Global Memory: " << globalMem / (1024*1024) << " MB" << std::endl;
    std::cout << "Local Memory: " << localMem / 1024 << " KB" << std::endl;
    std::cout << "Compute Units: " << computeUnits << std::endl;
    std::cout << "Max Work Group Size: " << maxWorkGroupSize << std::endl;
}

std::string OpenCLManager::getErrorString(cl_int error) {
    switch(error) {
        case CL_SUCCESS: return "Success";
        case CL_DEVICE_NOT_FOUND: return "Device not found";
        case CL_DEVICE_NOT_AVAILABLE: return "Device not available";
        case CL_COMPILER_NOT_AVAILABLE: return "Compiler not available";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "Memory object allocation failure";
        case CL_OUT_OF_RESOURCES: return "Out of resources";
        case CL_OUT_OF_HOST_MEMORY: return "Out of host memory";
        case CL_PROFILING_INFO_NOT_AVAILABLE: return "Profiling info not available";
        case CL_MEM_COPY_OVERLAP: return "Memory copy overlap";
        case CL_IMAGE_FORMAT_MISMATCH: return "Image format mismatch";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "Image format not supported";
        case CL_BUILD_PROGRAM_FAILURE: return "Build program failure";
        case CL_MAP_FAILURE: return "Map failure";
        case CL_INVALID_VALUE: return "Invalid value";
        case CL_INVALID_DEVICE_TYPE: return "Invalid device type";
        case CL_INVALID_PLATFORM: return "Invalid platform";
        case CL_INVALID_DEVICE: return "Invalid device";
        case CL_INVALID_CONTEXT: return "Invalid context";
        case CL_INVALID_QUEUE_PROPERTIES: return "Invalid queue properties";
        case CL_INVALID_COMMAND_QUEUE: return "Invalid command queue";
        case CL_INVALID_HOST_PTR: return "Invalid host pointer";
        case CL_INVALID_MEM_OBJECT: return "Invalid memory object";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "Invalid image format descriptor";
        case CL_INVALID_IMAGE_SIZE: return "Invalid image size";
        case CL_INVALID_SAMPLER: return "Invalid sampler";
        case CL_INVALID_BINARY: return "Invalid binary";
        case CL_INVALID_BUILD_OPTIONS: return "Invalid build options";
        case CL_INVALID_PROGRAM: return "Invalid program";
        case CL_INVALID_PROGRAM_EXECUTABLE: return "Invalid program executable";
        case CL_INVALID_KERNEL_NAME: return "Invalid kernel name";
        case CL_INVALID_KERNEL_DEFINITION: return "Invalid kernel definition";
        case CL_INVALID_KERNEL: return "Invalid kernel";
        case CL_INVALID_ARG_INDEX: return "Invalid argument index";
        case CL_INVALID_ARG_VALUE: return "Invalid argument value";
        case CL_INVALID_ARG_SIZE: return "Invalid argument size";
        case CL_INVALID_KERNEL_ARGS: return "Invalid kernel arguments";
        case CL_INVALID_WORK_DIMENSION: return "Invalid work dimension";
        case CL_INVALID_WORK_GROUP_SIZE: return "Invalid work group size";
        case CL_INVALID_WORK_ITEM_SIZE: return "Invalid work item size";
        case CL_INVALID_GLOBAL_OFFSET: return "Invalid global offset";
        case CL_INVALID_EVENT_WAIT_LIST: return "Invalid event wait list";
        case CL_INVALID_EVENT: return "Invalid event";
        case CL_INVALID_OPERATION: return "Invalid operation";
        case CL_INVALID_GL_OBJECT: return "Invalid GL object";
        case CL_INVALID_BUFFER_SIZE: return "Invalid buffer size";
        case CL_INVALID_MIP_LEVEL: return "Invalid mip level";
        case CL_INVALID_GLOBAL_WORK_SIZE: return "Invalid global work size";
        default: return "Unknown error";
    }
}