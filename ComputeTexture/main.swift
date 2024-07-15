//
//  main.swift
//  ComputeTexture
//
//  Created by larryhou on 2024/7/10.
//

import Foundation
import Metal
import MetalKit
import Cocoa

let COLOR_CHANNEL_THRESHOLD = 187
let SERVER_ADDRESS = "10.15.149.23:56789"
let THREAD_COUNT = 16

class Semaphore {
    var compute:[DispatchSemaphore] = []
    var loading:[DispatchSemaphore] = []
}

class Context {
    var device:MTLDevice
    var program:MTLComputePipelineState!
    var queue:[MTLCommandQueue] = []
    
    var semaphore:Semaphore = .init()
    
    var gputrace = false
    
    var lock:NSLock = .init()
    var id:String?
    
    init(device:MTLDevice) {
        self.device = device
    }
}

extension Int {
    static postfix func ++ (_ v: inout Int) -> Int {
        let t = v
        v += 1
        return t
    }
}

enum TextureError: Error {
    case notSupported
    case noMoreComputeJob
    case noTextureContent
}

struct TextureContext {
    var sizeX: Int
    var sizeY: Int
    var srgb: Int
    var format: Int
    var brightness:Float
    var path: String
    var texture:MTLTexture!
}

extension MTKTextureLoader {
    func newTexture(_ ctx: Context, semaphore:DispatchSemaphore) throws -> [TextureContext] {
        var data:Data?
        var filename:String?
        var components = URLComponents(string: "http://\(SERVER_ADDRESS)/compute")!
        components.queryItems = [
            .init(name: "id", value: ctx.id)
        ]
        
        var error:TextureError?
        let task = URLSession.shared.dataTask(with: .init(url: components.url!)) { raw, rsp, err in
            if let rsp = rsp as? HTTPURLResponse {
                if let group = rsp.value(forHTTPHeaderField: "Compute-Group"), ctx.id == nil {
                    ctx.lock.lock()
                    ctx.id = group
                    ctx.lock.unlock()
                }
                
                switch rsp.statusCode {
                case 410: 
                    error = .noMoreComputeJob
                case 200:
                    if let name = rsp.value(forHTTPHeaderField: "Compute-File") {
                        if !name.hasSuffix(".PNG") {
                            error = .notSupported
                        } else {
                            filename = name
                            data = raw
                        }
                    }
                    
                default:
                    break
                }
            }
            
            semaphore.signal()
        }

        task.resume()
        semaphore.wait()
        if let data = data {
            var textures:[TextureContext] = []
            var offset = 0
            while offset < data.count {
                let result = try newTexture(data, offset: offset)
                textures.append(result.0)
                offset = result.1
            }
            
            textures[0].path = filename!
            return textures
        }
        
        throw error ?? .noTextureContent
    }
    
    func newTexture(_ data:Data, offset:Int) throws -> (TextureContext, Int) {
        var ptr = data.withUnsafeBytes ({$0.baseAddress})!
        ptr = ptr.advanced(by: offset)
        
        let total = Int(ptr.load(as: Int32.self))
        ptr = ptr.advanced(by: 4)
        
        let size = Int(ptr.advanced(by: total-4).load(as: Int32.self))
        ptr = ptr.advanced(by: (size + 3) & ~3)
        
        var ctx:TextureContext = .init(sizeX: 0, sizeY: 0, srgb: 0, format: 0, brightness: 0, path: "")
        ctx.sizeX  = Int(ptr.advanced(by: 0).load(as: Int16.self))
        ctx.sizeY  = Int(ptr.advanced(by: 2).load(as: Int16.self))
        ctx.srgb   = Int(ptr.advanced(by: 4).load(as: Int16.self))
        ctx.format = Int(ptr.advanced(by: 6).load(as: Int16.self))
        ctx.brightness = ptr.advanced(by: 8).load(as: Float.self)
        
        let base = offset+4
        ctx.texture = try newTexture(data: data.subdata(in: base..<base+size), options: [
            .SRGB: NSNumber(value: 0),
            .textureUsage: NSNumber(value: MTLTextureUsage.shaderRead.rawValue),
            .textureStorageMode: NSNumber(value: MTLStorageMode.private.rawValue),
        ])
        return (ctx, offset + 4 + total)
    }
}

func draw(_ ctx:Context, queue:Int) throws {
    let fm = FileManager.default
    
    let loader = MTKTextureLoader(device: ctx.device)
    let textures = try loader.newTexture(ctx, semaphore: ctx.semaphore.loading[queue])
    let albedo = textures[0]
    let source = albedo.texture!
    
    print("LOAD \(albedo.path) \(String(describing: albedo.texture))")
    
    if ctx.gputrace {
        let cd = MTLCaptureDescriptor()
        cd.captureObject = ctx.device
        cd.destination = .gpuTraceDocument
        
        cd.outputURL = URL(fileURLWithPath: NSString(string: "~/Downloads/ComputeTexture.gputrace").expandingTildeInPath)
        if let url = cd.outputURL, fm.fileExists(atPath: url.path) { try fm.removeItem(at: url) }
        try MTLCaptureManager.shared().startCapture(with: cd)
    }
    
    defer {
        if ctx.gputrace {
            MTLCaptureManager.shared().stopCapture()
        }
    }
    guard let buffer = ctx.queue[queue].makeCommandBuffer() else {return}
    
    let td = MTLTextureDescriptor()
    td.textureType = source.textureType
    td.pixelFormat = .rgba8Unorm
    td.usage = [.shaderRead, .shaderWrite]
    td.mipmapLevelCount = source.mipmapLevelCount
    td.width = source.width
    td.height = source.height
    td.storageMode = .private
    
    guard let target = ctx.device.makeTexture(descriptor: td) else {return}
    
    let stats = ctx.device.makeBuffer(length: 128, options: [.storageModeManaged])!
    var uniform = Uniform(
        brightness: albedo.brightness,
        screen: simd_float2(Float(td.width), Float(td.height)),
        threshold: Int32(COLOR_CHANNEL_THRESHOLD),
        srgb: Int32(albedo.srgb)
    )
    
    let cpd = MTLComputePassDescriptor()
    cpd.dispatchType = .concurrent
    if let encoder = buffer.makeComputeCommandEncoder(descriptor: cpd) {
        let state = ctx.program!
        encoder.setComputePipelineState(state)
        encoder.setBytes(&uniform, length: MemoryLayout<Uniform>.stride, index: 0)
        encoder.setBuffer(stats, offset: 0, index: 1)
        
        var flag = textures.count > 1
        encoder.setBytes(&flag, length: MemoryLayout<Bool>.stride, index: 2)
        encoder.setTexture(textures.last?.texture, index: 2)
        
        for i in 0..<source.mipmapLevelCount {
            switch source.textureType {
            case .type2D:
                let src = source.makeTextureView(pixelFormat: source.pixelFormat, textureType: .type2D, levels: i..<i+1, slices: 0..<1)!
                let dst = target.makeTextureView(pixelFormat: target.pixelFormat, textureType: .type2D, levels: i..<i+1, slices: 0..<1)!
                encoder.setTexture(src, index: 0)
                encoder.setTexture(dst, index: 1)
                let size = MTLSize(width: state.threadExecutionWidth, height: state.maxTotalThreadsPerThreadgroup/state.threadExecutionWidth, depth: 1)
                let grid = MTLSize(width:  (src.width + size.width  - 1)/size.width ,
                                   height: (src.width + size.height - 1)/size.height, depth: 1)
                encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: size)
                
            case .typeCube:
                for j in 0..<6 {
                    let src = source.makeTextureView(pixelFormat: source.pixelFormat, textureType: .type2D, levels: i..<i+1, slices: j..<j+1)!
                    let dst = target.makeTextureView(pixelFormat: target.pixelFormat, textureType: .type2D, levels: i..<i+1, slices: j..<j+1)!
                    encoder.setTexture(src, index: 0)
                    encoder.setTexture(dst, index: 1)
                    let size = MTLSize(width: state.threadExecutionWidth, height: state.maxTotalThreadsPerThreadgroup/state.threadExecutionWidth, depth: 1)
                    let grid = MTLSize(width:  (src.width + size.width  - 1)/size.width ,
                                       height: (src.width + size.height - 1)/size.height, depth: 1)
                    encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: size)
                }
                
            default: break
            }
        }
        encoder.endEncoding()
    }
    
    let size = MTLSize(width: td.width, height: td.height, depth: 1)
    let image = ctx.device.makeBuffer(length: td.width * td.height * 4 * td.arrayLength)!
    if let encoder = buffer.makeBlitCommandEncoder() {
        encoder.synchronize(resource: stats)
        
        var offset = 0
        for _ in 0..<td.arrayLength {
            let bytesPerImage = size.width * size.height * 4
            encoder.copy(from: target, sourceSlice: 0, sourceLevel: 0, sourceOrigin: .init(), sourceSize: size,
                         to: image, destinationOffset: offset, destinationBytesPerRow: bytesPerImage/size.height, destinationBytesPerImage: bytesPerImage)
            offset += bytesPerImage
        }
        
        encoder.endEncoding()
    }
    
    buffer.addCompletedHandler { _ in
        source.setPurgeableState(.volatile)
        target.setPurgeableState(.volatile)
    }
    
    buffer.commit()
    buffer.waitUntilCompleted()
    
    let level = stats.contents().advanced(by: 0).load(as: Int32.self)
    let count = stats.contents().advanced(by: 4).load(as: Int32.self)
    if level >= uniform.threshold {
        var buffer:UnsafeMutablePointer<UInt8>? = image.contents().bindMemory(to: UInt8.self, capacity: image.length)
        let bitmap = NSBitmapImageRep(bitmapDataPlanes: &buffer,
                         pixelsWide: size.width,
                         pixelsHigh: size.height,
                         bitsPerSample: 8,
                         samplesPerPixel: 4,
                         hasAlpha: true,
                         isPlanar: false,
                         colorSpaceName: .deviceRGB,
                         bytesPerRow: size.width*4,
                         bitsPerPixel: 32)
        
        if let data = bitmap?.representation(using: .png, properties: [:]) {
            var components = URLComponents(string: "http://\(SERVER_ADDRESS)/compute")!
            components.queryItems = [
                .init(name: "path", value: albedo.path),
                .init(name: "id", value: ctx.id)
            ]
            
            var req = URLRequest(url: components.url!)
            req.httpMethod = "PUT"
            req.setValue("application/octet-stream", forHTTPHeaderField: "Content-Type")
            req.setValue("\(count)/\(td.width*td.height)", forHTTPHeaderField: "Compute-Ratio")
            req.httpBody = data
            URLSession.shared.dataTask(with: req, completionHandler: { _, rsp, _ in
                if let rsp = rsp as? HTTPURLResponse {
                    print("SEND \(albedo.path) \(rsp.statusCode)")
                }
            }).resume()
        }
        
        stats.setPurgeableState(.volatile)
        image.setPurgeableState(.volatile)
    }
}

if let device = MTLCreateSystemDefaultDevice() {
    let ctx = Context(device: device)
    if let library = device.makeDefaultLibrary() {
        ctx.program = try device.makeComputePipelineState(function: library.makeFunction(name: "compute")!)
    }
    
    for _ in 0..<THREAD_COUNT {
        ctx.queue.append(device.makeCommandQueue()!)
        ctx.semaphore.compute.append(.init(value: 0))
        ctx.semaphore.loading.append(.init(value: 0))
    }
    
    let queue = DispatchQueue(label: "Turbo", attributes: .concurrent)
    for (n, value) in CommandLine.arguments.enumerated() {
        if n > 0 {
            switch value {
            case "-gputrace": ctx.gputrace = true
            default: break
            }
        }
    }
    
    var compelete = false
    for i in 0..<THREAD_COUNT {
        let index = i
        queue.async {
            while !compelete {
                do { try draw(ctx, queue: index) }
                catch {
                    if let err = error as? TextureError, err == TextureError.noMoreComputeJob {
                        compelete = true
                    } else {
                        print("ERROR \(error)")
                    }
                }
            }
            ctx.semaphore.compute[index].signal()
        }
    }
    
    for i in 0..<ctx.semaphore.compute.count { ctx.semaphore.compute[i].wait() }
    
    var components = URLComponents(string: "http://\(SERVER_ADDRESS)/compute/summary")!
    components.queryItems = [.init(name: "id", value: ctx.id)]
    let url = components.url!
    
    let exit = ctx.semaphore.loading[0]
    URLSession.shared.dataTask(with: .init(url: url), completionHandler: { _, rsp, _ in
        if let rsp = rsp as? HTTPURLResponse {
            print("SUMMARY \(rsp.statusCode) \(url) \(rsp.allHeaderFields)")
        }
        exit.signal()
    }).resume()
    exit.wait()
}

