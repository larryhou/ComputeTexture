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

class Context {
    var device:MTLDevice
    var program:MTLComputePipelineState?
    var queue:[MTLCommandQueue] = []
    var semaphore:DispatchSemaphore?
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
    case noMoreTask
    case noImage
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
    func newTexture(_ ctx: Context) throws -> [TextureContext] {
        var data:Data?
        var filename:String?
        var components = URLComponents(string: "http://\(SERVER_ADDRESS)/compute")!
        components.queryItems = [
            .init(name: "id", value: ctx.id)
        ]
        
        var error:TextureError?
        let semaphore = DispatchSemaphore(value: 0)
        let task = URLSession.shared.dataTask(with: .init(url: components.url!)) { raw, rsp, err in
            if let rsp = rsp as? HTTPURLResponse {
                if let group = rsp.value(forHTTPHeaderField: "Compute-Group"), ctx.id == nil {
                    ctx.lock.lock()
                    ctx.id = group
                    ctx.lock.unlock()
                }
                
                switch rsp.statusCode {
                case 410: 
                    error = .noMoreTask
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
                    error = .noImage
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
        
        throw error ?? .noImage
    }
    
    func newTexture(_ data:Data, offset:Int) throws -> (TextureContext, Int) {
        var ptr = data.withUnsafeBytes ({$0}).baseAddress!
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
    let textures = try loader.newTexture(ctx)
    let albedo = textures[0]
    let source = albedo.texture!
    
    print("LOAD \(albedo.texture) \(source.description)")
    if ctx.gputrace {
        let cd = MTLCaptureDescriptor()
        cd.captureObject = ctx.device
        cd.destination = .gpuTraceDocument
        cd.outputURL = URL(fileURLWithPath: "/Users/larryhou/Downloads/ComputeTexture.gputrace")
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
    td.pixelFormat = .rgba16Float
    td.usage = [.shaderRead, .shaderWrite]
    td.mipmapLevelCount = source.mipmapLevelCount
    td.width = source.width
    td.height = source.height
    td.storageMode = .private
    
    guard let target = ctx.device.makeTexture(descriptor: td) else {return}
    
    let putback = ctx.device.makeBuffer(length: 128, options: [.storageModeManaged])!
    var uniform = Uniform(
        brightness: albedo.brightness,
        screen: simd_float2(Float(td.width), Float(td.height)),
        threshold: Int32(COLOR_CHANNEL_THRESHOLD),
        srgb: Int32(albedo.srgb)
    )
    
    let cpd = MTLComputePassDescriptor()
    cpd.dispatchType = .concurrent
    if let encoder = buffer.makeComputeCommandEncoder(descriptor: cpd), let state = ctx.program {
        encoder.setComputePipelineState(state)
        encoder.setBytes(&uniform, length: MemoryLayout<Uniform>.stride, index: 0)
        encoder.setBuffer(putback, offset: 0, index: 1)
        
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
    
    if let encoder = buffer.makeBlitCommandEncoder() {
        encoder.synchronize(resource: putback)
        encoder.endEncoding()
    }
    
    buffer.addCompletedHandler { _ in
        source.setPurgeableState(.volatile)
    }
    
    buffer.commit()
    buffer.waitUntilCompleted()
    
    let level = putback.contents().advanced(by: 0).load(as: Int32.self)
    let count = putback.contents().advanced(by: 4).load(as: Int32.self)
    if level >= uniform.threshold {
        if let image = CIImage(mtlTexture: target, options: [.colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!]) {
            let bitmap = NSBitmapImageRep(ciImage: image.oriented(.downMirrored))
            if let data = bitmap.representation(using: .png, properties: [:]) {
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
                
                target.setPurgeableState(.volatile)
                putback.setPurgeableState(.volatile)
            }
        }
    }
}

if let device = MTLCreateSystemDefaultDevice() {
    let ctx = Context(device: device)
    if let library = device.makeDefaultLibrary() {
        ctx.program = try device.makeComputePipelineState(function: library.makeFunction(name: "compute")!)
    }
    
    let limit = 8
    for _ in 0..<limit {
        if let queue = device.makeCommandQueue() { ctx.queue.append(queue) }
    }
    
    ctx.semaphore = .init(value: limit)
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
    
    var i = 0
    while !compelete || i == 8 {
        ctx.semaphore?.wait()
        let index = i % limit
        queue.async {
            do { try draw(ctx, queue: index) }
            catch {
                if let err = error as? TextureError, err == TextureError.noMoreTask {
                    compelete = true
                } else {
                    print("ERROR \(error)")
                }
            }
            ctx.semaphore?.signal()
        }
        
        i = i + 1
    }
    
    for _ in 0..<limit { ctx.semaphore?.wait()   }
    for _ in 0..<limit { ctx.semaphore?.signal() }
    
    Thread.sleep(until: .now.addingTimeInterval(10))
    
    
    var components = URLComponents(string: "http://\(SERVER_ADDRESS)/compute/summary")!
    components.queryItems = [
        .init(name: "id", value: ctx.id)
    ]
    let url = components.url!
    print("\(url)")
    URLSession.shared.dataTask(with: .init(url: url)).resume()
    Thread.sleep(until: .now.addingTimeInterval(1))
}

