import './style.css'
import computeWGSL from './compute.wgsl?raw';


let errorReported = false;
let device: GPUDevice | null = null;
let sgemmPipeline: GPUComputePipeline | null = null;
let sgemmBindGroupLayout: GPUBindGroupLayout | null = null;

// Initialize WebGPU - moved to an async function
async function initWebGPU(): Promise<void> {
  if (!navigator.gpu) throw new Error('WebGPU not available: navigator.gpu is missing.');
  
  const adapter = await navigator.gpu.requestAdapter({
    featureLevel: 'compatibility',
  });
  if (!adapter) throw new Error('WebGPU not available: requestAdapter() returned null.');
  
  device = await adapter.requestDevice();
  if (!device) throw new Error('WebGPU not available: requestDevice() returned null.');

  // Initialize compute pipeline
  await initSGEMMPipeline();
}

// Initialize the compute pipeline
async function initSGEMMPipeline(): Promise<void> {
  if (!device) throw new Error('WebGPU device not available');

  try {
    const shaderModule = device.createShaderModule({
      code: computeWGSL,
    });

    // Compile the shader module to check for errors
    const compilationInfo = await shaderModule.getCompilationInfo();
    if (compilationInfo.messages.length > 0) {
      console.warn('Shader compilation messages:', compilationInfo.messages);
    }

    // Create bind group layout
    sgemmBindGroupLayout = device.createBindGroupLayout({
      entries: [
        {
          binding: 0, // m, n, k, alpha
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'uniform' }
        },
        {
          binding: 1, // matrix A
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'read-only-storage' }
        },
        {
          binding: 2, // matrix B
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'read-only-storage' }
        },
        {
          binding: 3, // matrix C (result)
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'storage' }
        }
      ]
    });

    // Create pipeline layout
    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [sgemmBindGroupLayout]
    });

    // Create compute pipeline
    sgemmPipeline = device.createComputePipeline({
      layout: pipelineLayout,
      compute: {
        module: shaderModule,
        entryPoint: 'main',
      }
    });
  } catch (error) {
    console.error('Failed to create compute pipeline:', error);
    throw error;
  }
}

// Initialize WebGPU when the page loads
initWebGPU().catch(console.error);


function alertIfError() {
  if (errorReported) {
    return;
  }

  let error = false; //getWebGPUError();
  if (error) {
    alert(`WebGPU Error (fallback to pure JavaScript): ${error}`);
    errorReported = true;
  }
}

function message(m: string, target: string): void {
  document.getElementById(target).innerText += m + '\n';
}

function makeRandom(length: number): Float32Array {
  const array = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    array[i] = Math.random();
  }

  return array;
}

function checkResult(m: number, n: number, k: number, alpha: number, array_a: Float32Array, array_b: Float32Array, actual: Float32Array): boolean {
  const expected = new Float32Array(m * n);
  for (let row = 0; row < m; row++) {
    for (let col = 0; col < n; col++) {
      let sum = 0.0;
      for (let j = 0; j < k; j++) {
        sum += array_a[row * k + j] * array_b[j * n + col];
      }
      expected[row * n + col] = sum * alpha;
    }
  }
  for (let row = 0; row < m; row++) {
    for (let col = 0; col < n; col++) {
      const idx = row * n + col;
      const expected_el = expected[idx];
      const actual_el = actual[idx];
      if (Math.abs(expected_el - actual_el) > (1e-5 + 1e-3 * Math.abs(expected_el))) {
        console.error(`[${row}, ${col}]: ${expected_el} !== ${actual_el}`);
        return false;
      }
    }
  }
  return true;
}

// packing buffer into a vec4 of floats instead of an array
function packRowMajorToVec4(src: Float32Array, rows: number, cols: number) {
  const cols4 = Math.ceil(cols / 4);          // number of vec4s per row
  const packed = new Float32Array(rows * cols4 * 4); // padded storage in floats
  for (let r = 0; r < rows; r++) {
    packed.set(
      src.subarray(r * cols, r * cols + cols),
      r * cols4 * 4
    );
  }
  return { packed, cols4 };
}

function unpackRowMajorFromVec4(packed: Float32Array, rows: number, cols: number, cols4: number) {
  const out = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    out.set(
      packed.subarray(r * cols4 * 4, r * cols4 * 4 + cols),
      r * cols
    );
  }
  return out;
}

function align16(bytes: number) {
  return (bytes + 15) & ~15;
}


function parseMNKTuples(s: string): number[][] {
  const shapes: number[][] = [];//[[m,n,k]]
  for (const line of s.split('\n')) {
    const parts = line.split(',').map((t) => Number(t.trim()));
    if (parts.length === 3 && parts.every((v) => v > 0)) {
      shapes.push(parts);
    }
  }
  return shapes;
}

async function run_benchmark() {
  const messageTarget = 'bench_message';
  try {
    const shapes = parseMNKTuples(
      (document.getElementById('benchmark_shapes') as HTMLTextAreaElement).value
    );
    const alpha = 1;
    const runs = 10;
    const warmups = 3; // do multiple warmups
    for (const [m, n, k] of shapes) {
      const array_a = makeRandom(m * k);
      const array_b = makeRandom(k * n);

      // Warmup runs (not measured) to stabilize driver/JIT/GPU clocks
      for (let w = 0; w < warmups; w++) {
        const warm = await sgemm(m, n, k, alpha, array_a, array_b);
        // Force GPU completion if sgemm doesn't already do it:
        if (typeof device !== 'undefined' && device.queue && device.queue.onSubmittedWorkDone) {
          await device.queue.onSubmittedWorkDone();
        }
      }

      // measured runs
      let timeSum = 0;
      let checksum = 0;
      for (let i = 0; i < runs; i++) {
        const t0 = performance.now();
        const result = await sgemm(m, n, k, alpha, array_a, array_b);

        // Ensure GPU actually finished. If your sgemm already waits (via mapAsync),
        // remove the following line. If not, this ensures synchronization.
        if (typeof device !== 'undefined' && device.queue && device.queue.onSubmittedWorkDone) {
          await device.queue.onSubmittedWorkDone();
        }

        const t1 = performance.now();
        timeSum += (t1 - t0);

        // cheap checksum over returned buffer to avoid dead-code elimination and
        // to detect correctness regressions. Sum a handful of elements instead of only one.
        // If result is a typed array:
        if (result && result.length) {
          // sample up to first 16 elements
          const sampleCount = Math.min(16, result.length);
          let s = 0;
          for (let si = 0; si < sampleCount; si++) s += result[si];
          checksum += s;
        } else {
          checksum += Number(result || 0);
        }
      }

      const avgTimeMs = timeSum / runs;
      // GFLOPS = (2 * m * n * k) / (avgTimeSeconds) / 1e9
      const avgTimeSec = avgTimeMs / 1000.0;
      const gflops = (2.0 * m * n * k) / (avgTimeSec * 1e9);

      message(`Sgemm (${m}x${k}) * (${k}x${n}): avg ${avgTimeMs.toFixed(3)} ms over ${runs} runs — ${gflops.toFixed(2)} GFLOPS/s`, messageTarget);
      console.log('checksum (to avoid optimization):', checksum);
    }
  } catch (ex) {
    // avoid blocking alert in hot loops; log instead
    console.error('Benchmark error:', ex);
    alert(ex && ex.message ? ex.message : String(ex));
  }
}

async function small_example() {
  console.log('running tezy');
  try {
    const array_a = new Float32Array([1, 2, 3, 4]);
    const array_b = new Float32Array([5, 6, 7, 8]);
    const result = await sgemm(2, 2, 2, 1, array_a, array_b);
    alertIfError();
    document.getElementById('small_example_result').innerText = `[${result[0]}, ${result[1]}\n ${result[2]}, ${result[3]}]`;
  } catch (ex) {
    alert(ex.message);
  }
}

async function run_test() {
  const shapes = parseMNKTuples((document.getElementById('test_shapes') as HTMLTextAreaElement).value)
  const alpha = 1; //1.0;
  const messageTarget = 'test_message';
  for (const [m, n, k] of shapes) {
    const array_a = makeRandom(m * k);
    const array_b = makeRandom(k * n);
    const result = await sgemm(m, n, k, alpha, array_a, array_b);
    alertIfError();
    const validation_result = checkResult(m, n, k, alpha, array_a, array_b, result);
    message(`M=${m}, N=${n}, K=${k}: ${validation_result ? 'OK' : 'Error'}`, messageTarget);
  }
}

window.addEventListener('load', () => {
  document.getElementById('run_benchmark').onclick = run_benchmark;
  document.getElementById('small_example').onclick = small_example;
  document.getElementById('run_test').onclick = run_test;
  document.getElementById('is_webgpu_enabled').innerText = (navigator as any).gpu ? 'Enabled' : 'Disabled (fallback pure JavaScript implementation will be used)';
});



function assert(cond: boolean, msg = '') {
  if (!cond) {
    throw new Error(msg);
  }
}



// Replace the sgemm function with this corrected version
async function sgemm(m: number, n: number, k: number, alpha: number, array_a: Float32Array, array_b: Float32Array): Promise<Float32Array> {
  
  if (!device) {
    throw new Error('WebGPU device not available');
  }
  
  // Create buffers - we need 5 buffers total:
  // 1. Uniform buffer (m, n, k, alpha)
  // 2. Input buffer A (matrix A data)
  // 3. Input buffer B (matrix B data) 
  // 4. Output buffer (result matrix C on GPU)
  // 5. Readback buffer (to copy results back to CPU)

  // vector ops
  // const {packed: aPacked, cols4: KD4} = packRowMajorToVec4(array_a, m, k);
  // const {packed: bPacked, cols4: ND4} = packRowMajorToVec4(array_b, k ,n);

   // C is m rows, ND4 vec4s per row => m * ND4 vec4s => bytes = *16
  // const cPackedFloatCount = m * ND4 * 4;                 // float count in packed C
  // const resultBufferSize = cPackedFloatCount * 4;   

  // float ops
  const aBufferSize = array_a.byteLength;
  const bBufferSize = array_b.byteLength;
  const resultBufferSize = m * n * Float32Array.BYTES_PER_ELEMENT;

  //const uniformBufferSize = 7 * Float32Array.BYTES_PER_ELEMENT; // m, n, k, alpha as f32
  const uniformBufferSize = 4 * Float32Array.BYTES_PER_ELEMENT;

  // 1. Uniform buffer - holds the matrix dimensions and alpha
  const uniformBuffer = device.createBuffer({
    size: uniformBufferSize,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // 2. Input buffer A - matrix A data (read-only for shader)
  const aBuffer = device.createBuffer({
    size: aBufferSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  // 3. Input buffer B - matrix B data (read-only for shader)
  const bBuffer = device.createBuffer({
    //size: bBufferSize,
    //size: bPacked.byteLength,
    size: bBufferSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  // 4. Output buffer - result matrix C (read-write for shader)
  const resultBuffer = device.createBuffer({
    size: resultBufferSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  // 5. Readback buffer - used to copy results back to CPU
  const readbackBuffer = device.createBuffer({
    size: resultBufferSize,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  // Write data to buffers
  // Uniform data: [m, n, k, alpha] all as floats
  const uniformData = new Int32Array(4);
  uniformData[0] = m >>> 0;
  uniformData[1] = n >>> 0;
  uniformData[2] = k >>> 0;
  uniformData[3] = alpha >>> 0;

  device.queue.writeBuffer(uniformBuffer, 0, uniformData);
  //device.queue.writeBuffer(aBuffer, 0, aPacked);
  device.queue.writeBuffer(bBuffer, 0, array_b);
  device.queue.writeBuffer(aBuffer, 0, array_a);
  //device.queue.writeBuffer(bBuffer, 0, bPacked);
  // Create bind group - connects our buffers to the shader
  const bindGroup = device.createBindGroup({
    layout: sgemmBindGroupLayout,
    entries: [
      {
        binding: 0, // uniform buffer (m, n, k, alpha)
        resource: { buffer: uniformBuffer }
      },
      {
        binding: 1, // matrix A
        resource: { buffer: aBuffer }
      },
      {
        binding: 2, // matrix B  
        resource: { buffer: bBuffer }
      },
      {
        binding: 3, // result matrix C
        resource: { buffer: resultBuffer }
      }
    ]
  });

  // Create command encoder and compute pass
  const commandEncoder = device.createCommandEncoder();
  const computePass = commandEncoder.beginComputePass();

  computePass.setPipeline(sgemmPipeline);
  computePass.setBindGroup(0, bindGroup);
  
  // Calculate workgroup counts - adjust based on your compute.wgsl workgroup size
  // Typical workgroup size is 8x8 or 16x16 threads
  const workgroupSizeX = 16; 
  const workgroupSizeY = 16; 
  const workgroupsX = Math.ceil(n / workgroupSizeX);
  const workgroupsY = Math.ceil(m / workgroupSizeY);
  
  computePass.dispatchWorkgroups(workgroupsX, workgroupsY, 1);
  computePass.end();

  // Copy result from GPU storage buffer to CPU-readable buffer
  commandEncoder.copyBufferToBuffer(resultBuffer, 0, readbackBuffer, 0, resultBufferSize);

  // Submit commands to GPU
  const commands = commandEncoder.finish();
  device.queue.submit([commands]);

  // Read back results from GPU to CPU
  await device.queue.onSubmittedWorkDone();
  await readbackBuffer.mapAsync(GPUMapMode.READ);
  const resultArray = new Float32Array(readbackBuffer.getMappedRange().slice(0));
  //const mapped = readbackBuffer.getMappedRange();
  //const cPacked = new Float32Array(mapped.slice(0));
  readbackBuffer.unmap();

  // Unpack packed C back to normal m*n floats for the checker
  //const result = unpackRowMajorFromVec4(cPacked, m, n, ND4);

  // Clean up buffers (optional but good practice)
  uniformBuffer.destroy();
  aBuffer.destroy();
  bBuffer.destroy();
  resultBuffer.destroy();
  readbackBuffer.destroy();

  //return resultArray;
  return resultArray;
}
