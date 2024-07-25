import Metal
import MetalPerformanceShaders
import Accelerate
import Foundation

let metallib = "\(#file.replacingOccurrences(of: "/PyMetalBridge.swift", with: ""))/Shaders.metallib"

@available(macOS 10.13, *)
let device = MTLCreateSystemDefaultDevice()!,
    commandQueue = device.makeCommandQueue()!,
    defaultLibrary = try! device.makeLibrary(filepath: metallib)

@available(macOS 10.13, *)
@_cdecl("swift_matrix_multiplication_on_gpu")
public func swift_matrix_multiplication_on_gpu(a: UnsafePointer<Float>, b: UnsafePointer<Float>, result: UnsafeMutablePointer<Float>, M: Int, N: Int, K: Int) -> Int {
    return computeMatrixMultiplicationOnGPU(a: a, b: b, result: result, M: M, N: N, K: K)
}

@available(macOS 10.13, *)
func computeMatrixMultiplicationOnGPU(a: UnsafePointer<Float>, b: UnsafePointer<Float>, result: UnsafeMutablePointer<Float>, M: Int, N: Int, K: Int) -> Int {
    do {
        let commandBuffer = commandQueue.makeCommandBuffer()!
        let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()!

        let function = defaultLibrary.makeFunction(name: "matrix_multiplication")!
        let computePipelineState = try device.makeComputePipelineState(function: function)
        computeCommandEncoder.setComputePipelineState(computePipelineState)

        let bufferSizeA = M * K * MemoryLayout<Float>.size
        let bufferSizeB = K * N * MemoryLayout<Float>.size
        let bufferSizeResult = M * N * MemoryLayout<Float>.size

        let bufferA = device.makeBuffer(bytes: a, length: bufferSizeA, options: [])
        let bufferB = device.makeBuffer(bytes: b, length: bufferSizeB, options: [])
        let bufferResult = device.makeBuffer(length: bufferSizeResult, options: [])

        computeCommandEncoder.setBuffer(bufferA, offset: 0, index: 0)
        computeCommandEncoder.setBuffer(bufferB, offset: 0, index: 1)
        computeCommandEncoder.setBuffer(bufferResult, offset: 0, index: 2)

        var M = M, N = N, K = K
        computeCommandEncoder.setBytes(&M, length: MemoryLayout<Int>.size, index: 3)
        computeCommandEncoder.setBytes(&N, length: MemoryLayout<Int>.size, index: 4)
        computeCommandEncoder.setBytes(&K, length: MemoryLayout<Int>.size, index: 5)

        let threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let threadgroupCount = MTLSize(width: (M + 15) / 16, height: (N + 15) / 16, depth: 1)

        computeCommandEncoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadgroupSize)
        computeCommandEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let data = Data(bytesNoCopy: bufferResult!.contents(), count: bufferSizeResult, deallocator: .none)
        data.copyBytes(to: UnsafeMutableRawPointer(result).assumingMemoryBound(to: UInt8.self), count: bufferSizeResult)

        return 0
    } catch {
        print("\(error)")
        return 1
    }
}
