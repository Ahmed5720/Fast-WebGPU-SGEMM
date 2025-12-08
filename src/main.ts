import './style.css'
import computeWGSL from './compute.wgsl?raw';


const workgroupSizeX = 16;
const workgroupSizeY = 16;

let errorReported = false;
let device: GPUDevice | null = null;
let sgemmNaivePipeline: GPUComputePipeline | null = null;
let sgemmNaiveCoalescedPipeline: GPUComputePipeline | null = null;
let sgemmBlockedPipeline: GPUComputePipeline | null = null;
let sgemmVectorizedPipeline: GPUComputePipeline | null = null;


let sgemmBindGroupLayout: GPUBindGroupLayout | null = null;




let runtime = {start_t:0, end_t:0};
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

    // Create compute pipelines
    sgemmNaivePipeline = device.createComputePipeline({
      layout: pipelineLayout,
      compute: {
        module: shaderModule,
        entryPoint: 'naive',
      }
    });

    // Create compute pipelines
    sgemmNaiveCoalescedPipeline = device.createComputePipeline({
      layout: pipelineLayout,
      compute: {
        module: shaderModule,
        entryPoint: 'naive_mem_coalesced',
      }
    });

    // Create compute pipelines
    sgemmBlockedPipeline = device.createComputePipeline({
      layout: pipelineLayout,
      compute: {
        module: shaderModule,
        entryPoint: 'block_tiling',
      }
    });

    // Create compute pipelines
    sgemmVectorizedPipeline = device.createComputePipeline({
      layout: pipelineLayout,
      compute: {
        module: shaderModule,
        entryPoint: 'block_tiling_vectorized',
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
// Store results globally for plotting
let benchmarkResults = {
  sizes: [],
  sgemm1_gflops: [],
  sgemm2_gflops: [],
  sgemm3_gflops: [],
  sgemm4_gflops: []
};

async function run_all_benchmarks() {
  const messageTarget = 'bench_message';
  console.log('zeby');
  try {
    // Clear previous results
    benchmarkResults = {
      sizes: [],
      sgemm1_gflops: [],
      sgemm2_gflops: [],
      sgemm3_gflops: [],
      sgemm4_gflops: []
    };
    
    const shapes = parseMNKTuples(
      (document.getElementById('benchmark_shapes') as HTMLTextAreaElement).value
    );
    const alpha = 1;
    const runs = 10;
    const warmups = 3;
    
    // Create arrays for each kernel's results
    const sgemmFunctions = [sgemm1, sgemm2, sgemm3, sgemm4];
    const sgemmNames = ['sgemm1', 'sgemm2', 'sgemm3', 'sgemm4'];
    
    for (const [m, n, k] of shapes) {
      const array_a = makeRandom(m * k);
      const array_b = makeRandom(k * n);
      
      // Store matrix size for plotting
      benchmarkResults.sizes.push(`${m}x${n}x${k}`);
      
      // Run each sgemm kernel
      for (let kernelIdx = 0; kernelIdx < sgemmFunctions.length; kernelIdx++) {
        const sgemmFunc = sgemmFunctions[kernelIdx];
        const kernelName = sgemmNames[kernelIdx];
        
        message(`Running ${kernelName} for size ${m}x${n}x${k}...`, messageTarget);
        
        // Warmup runs
        for (let w = 0; w < warmups; w++) {
          const warm = await sgemmFunc(m, n, k, alpha, array_a, array_b, runtime);
          if (typeof device !== 'undefined' && device.queue && device.queue.onSubmittedWorkDone) {
            await device.queue.onSubmittedWorkDone();
          }
        }
        
        // Measured runs
        let timeSum = 0;
        let checksum = 0;
        for (let i = 0; i < runs; i++) {
          //const t0 = performance.now();
          runtime = {start_t: 0, end_t: 0};
          const result = await sgemmFunc(m, n, k, alpha, array_a, array_b, runtime);
          
          if (typeof device !== 'undefined' && device.queue && device.queue.onSubmittedWorkDone) {
            await device.queue.onSubmittedWorkDone();
          }
          
          //const t1 = performance.now();
          timeSum += (runtime.end_t - runtime.start_t);
          
          // Checksum
          if (result && result.length) {
            const sampleCount = Math.min(16, result.length);
            let s = 0;
            for (let si = 0; si < sampleCount; si++) s += result[si];
            checksum += s;
          } else {
            checksum += Number(result || 0);
          }
        }
        
        const avgTimeMs = timeSum / runs;
        const avgTimeSec = avgTimeMs / 1000.0;
        const gflops = (2.0 * m * n * k) / (avgTimeSec * 1e9);
        
        // Store result for plotting
        switch(kernelIdx) {
          case 0:
            benchmarkResults.sgemm1_gflops.push(gflops);
            break;
          case 1:
            benchmarkResults.sgemm2_gflops.push(gflops);
            break;
          case 2:
            benchmarkResults.sgemm3_gflops.push(gflops);
            break;
          case 3:
            benchmarkResults.sgemm4_gflops.push(gflops);
            break;
        }
        
        message(`${kernelName} (${m}x${k}) * (${k}x${n}): avg ${avgTimeMs.toFixed(3)} ms — ${gflops.toFixed(2)} GFLOPS/s`, messageTarget);
        console.log(`${kernelName} checksum:`, checksum);
      }
      
      // Add spacing between different matrix sizes
      message('---', messageTarget);
    }
    
    // Create plot after all benchmarks are done
    plotBenchmarkResults();
    
  } catch (ex) {
    console.error('Benchmark error:', ex);
    alert(ex && ex.message ? ex.message : String(ex));
  }
}

// Function to plot benchmark results
function plotBenchmarkResults() {
  const canvas = document.getElementById('benchmarkPlot');
  if (!canvas) {
    console.error('Canvas element not found for plotting');
    return;
  }
  
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    console.error('Could not get canvas context');
    return;
  }
  
  // Clear canvas
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  // Plot configuration
  const padding = 60;
  const plotWidth = canvas.width - 2 * padding;
  const plotHeight = canvas.height - 2 * padding;
  
  // Find min and max values for scaling
  const allGflops = [
    ...benchmarkResults.sgemm1_gflops,
    ...benchmarkResults.sgemm2_gflops,
    ...benchmarkResults.sgemm3_gflops,
    ...benchmarkResults.sgemm4_gflops
  ];
  
  const maxGflops = Math.max(...allGflops);
  const minGflops = Math.min(...allGflops);
  
  // Add some padding to Y axis
  const yMax = maxGflops * 1.1;
  const yMin = Math.max(0, minGflops * 0.9);
  
  // Colors for different kernels
  const colors = [
    '#ff4242ff', // Red
    '#b414d4ff', // Teal
    '#45B7D1', // Blue
    '#0da960ff'  // Green
  ];
  
  const kernelNames = ['Naive', 'Naive - Coalesced', 'Coalesced & Block Tiling', 'Coalesced & BlockTiled & vectorized'];
  
  // Draw axes
  ctx.strokeStyle = '#333';
  ctx.lineWidth = 2;
  
  // X-axis
  ctx.beginPath();
  ctx.moveTo(padding, canvas.height - padding);
  ctx.lineTo(canvas.width - padding, canvas.height - padding);
  ctx.stroke();
  
  // Y-axis
  ctx.beginPath();
  ctx.moveTo(padding, padding);
  ctx.lineTo(padding, canvas.height - padding);
  ctx.stroke();
  
  // Draw grid lines
  ctx.strokeStyle = '#ddd';
  ctx.lineWidth = 1;
  
  // X grid (vertical)
  const xStep = plotWidth / (benchmarkResults.sizes.length - 1);
  for (let i = 0; i < benchmarkResults.sizes.length; i++) {
    const x = padding + i * xStep;
    ctx.beginPath();
    ctx.moveTo(x, padding);
    ctx.lineTo(x, canvas.height - padding);
    ctx.stroke();
  }
  
  // Y grid (horizontal)
  const numYTicks = 10;
  for (let i = 0; i <= numYTicks; i++) {
    const y = canvas.height - padding - (i * plotHeight / numYTicks);
    ctx.beginPath();
    ctx.moveTo(padding, y);
    ctx.lineTo(canvas.width - padding, y);
    ctx.stroke();
  }
  
  // Draw Y-axis labels
  ctx.fillStyle = '#333';
  ctx.font = '12px Arial';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  
  for (let i = 0; i <= numYTicks; i++) {
    const value = yMin + (i * (yMax - yMin) / numYTicks);
    const y = canvas.height - padding - (i * plotHeight / numYTicks);
    ctx.fillText(value.toFixed(0), padding - 10, y);
  }
  
  // Draw X-axis labels (matrix sizes)
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (let i = 0; i < benchmarkResults.sizes.length; i++) {
    const x = padding + i * xStep;
    //ctx.fillText(benchmarkResults.sizes[i], x, canvas.height - padding + 10);
    
    // Rotate text if too many labels
    if (benchmarkResults.sizes.length > 5) {
      ctx.save();
      ctx.translate(x, canvas.height - padding + 20);
      ctx.rotate(-Math.PI / 4);
      ctx.fillText(benchmarkResults.sizes[i], 0, 0);
      ctx.restore();
    }
  }
  
 for (let kernelIdx = 0; kernelIdx < 4; kernelIdx++) {
    const gflopsArray = benchmarkResults[`sgemm${kernelIdx + 1}_gflops`];
    
    if (!gflopsArray || gflopsArray.length === 0) continue;
    
    // Draw connecting lines
    ctx.strokeStyle = colors[kernelIdx];
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    for (let i = 0; i < gflopsArray.length; i++) {
      const x = padding + i * xStep;
      const y = canvas.height - padding - 
                ((gflopsArray[i] - yMin) / (yMax - yMin)) * plotHeight;
      
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();
    
    // Draw data points on top of the lines
    ctx.fillStyle = colors[kernelIdx];
    for (let i = 0; i < gflopsArray.length; i++) {
      const x = padding + i * xStep;
      const y = canvas.height - padding - 
                ((gflopsArray[i] - yMin) / (yMax - yMin)) * plotHeight;
      
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, Math.PI * 2);
      ctx.fill();
      
      // Optional: Add a white border to make points stand out
      ctx.strokeStyle = 'white';
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.strokeStyle = colors[kernelIdx]; // Reset to original color
    }
  }
  
  // Add legend
  const legendX = canvas.width - 150;
  const legendY = padding;
  
  ctx.font = 'bold 14px Arial';
  ctx.fillStyle = '#333';
  ctx.textAlign = 'left';
  ctx.fillText('SGEMM Kernels', legendX, legendY - 25);
  
  ctx.font = '12px Arial';
  for (let i = 0; i < 4; i++) {
    const y = legendY + i * 25;
    
    ctx.fillStyle = colors[i];
    ctx.beginPath();
    ctx.rect(legendX, y - 8, 15, 15);
    ctx.fill();
    
    ctx.fillStyle = '#333';
    ctx.fillText(kernelNames[i], legendX + 25, y);
  }
  
  // Add axis labels
  ctx.font = 'bold 14px Arial';
  ctx.textAlign = 'center';
  ctx.fillText('Matrix Size (m×n×k)', canvas.width / 2, canvas.height - 15);
  
  ctx.save();
  ctx.translate(20, canvas.height / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('Performance (GFLOPS/s)', 0, 0);
  ctx.restore();
  
  // Add title
  ctx.font = 'bold 16px Arial';
  ctx.fillStyle = '#333';
  ctx.textAlign = 'center';
  ctx.fillText('SGEMM Kernel Performance Comparison', canvas.width / 2, 20);
}

// Helper function to display the plot in the UI
function displayPlot() {
  const container = document.getElementById('plotContainer');
  if (!container) {
    // Create plot container if it doesn't exist
    const newContainer = document.createElement('div');
    newContainer.id = 'plotContainer';
    newContainer.innerHTML = `
      <h3>Performance Results</h3>
      <canvas id="benchmarkPlot" width="800" height="500" style="border: 1px solid #ddd; background: white;"></canvas>
      <button onclick="plotBenchmarkResults()" style="margin-top: 10px;">Refresh Plot</button>
    `;
    document.body.appendChild(newContainer);
  }
  
  // Ensure canvas exists and plot results
  plotBenchmarkResults();
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
        const warm = await sgemm1(m, n, k, alpha, array_a, array_b);
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
        const result = await sgemm1(m, n, k, alpha, array_a, array_b);

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
      console.log('checksum', checksum);
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
    const result = await sgemm1(2, 2, 2, 1, array_a, array_b, runtime);
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
    const result = await sgemm2(m, n, k, alpha, array_a, array_b, runtime);
    alertIfError();
    const validation_result = checkResult(m, n, k, alpha, array_a, array_b, result);
    message(`M=${m}, N=${n}, K=${k}: ${validation_result ? 'OK' : 'Error'}`, messageTarget);
  }
}

window.addEventListener('load', () => {
  document.getElementById('run_benchmark').onclick = run_all_benchmarks;
  document.getElementById('small_example').onclick = small_example;
  document.getElementById('run_test').onclick = run_test;
  document.getElementById('is_webgpu_enabled').innerText = (navigator as any).gpu ? 'Enabled' : 'Disabled (fallback pure JavaScript implementation will be used)';
});



function assert(cond: boolean, msg = '') {
  if (!cond) {
    throw new Error(msg);
  }
}


async function sgemm1(m: number, n: number, k: number, alpha: number, array_a: Float32Array, array_b: Float32Array, runtime): Promise<Float32Array> {
  
  if (!device) {
    throw new Error('WebGPU device not available');
  }
  
  // Create buffers - we need 5 buffers total:
  // 1. Uniform buffer (m, n, k, alpha)
  // 2. Input buffer A (matrix A data)
  // 3. Input buffer B (matrix B data) 
  // 4. Output buffer (result matrix C on GPU)
  // 5. Readback buffer (to copy results back to CPU)

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
  device.queue.writeBuffer(bBuffer, 0, array_b);
  device.queue.writeBuffer(aBuffer, 0, array_a);
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

  computePass.setPipeline(sgemmNaivePipeline);
  computePass.setBindGroup(0, bindGroup);
  
  // Calculate workgroup counts - adjust based on your compute.wgsl workgroup size
  // Typical workgroup size is 8x8 or 16x16 threads
  
  const workgroupsX = Math.ceil(n / workgroupSizeX);
  const workgroupsY = Math.ceil(m / workgroupSizeY);
  runtime.start_t = performance.now();
  computePass.dispatchWorkgroups(workgroupsX, workgroupsY, 1);
  computePass.end();

  // Copy result from GPU storage buffer to CPU-readable buffer
  commandEncoder.copyBufferToBuffer(resultBuffer, 0, readbackBuffer, 0, resultBufferSize);

  // Submit commands to GPU
  let commands = commandEncoder.finish();
  device.queue.submit([commands]);


  // Read back results from GPU to CPU
  await device.queue.onSubmittedWorkDone();

  runtime.end_t = performance.now();

  await readbackBuffer.mapAsync(GPUMapMode.READ);
  const resultArray = new Float32Array(readbackBuffer.getMappedRange().slice(0));
  //const mapped = readbackBuffer.getMappedRange();
  //const cPacked = new Float32Array(mapped.slice(0));
  readbackBuffer.unmap();


  // Clean up buffers (optional but good practice)
  uniformBuffer.destroy();
  aBuffer.destroy();
  bBuffer.destroy();
  resultBuffer.destroy();
  readbackBuffer.destroy();

  return resultArray;
}



async function sgemm2(m: number, n: number, k: number, alpha: number, array_a: Float32Array, array_b: Float32Array, runtime): Promise<Float32Array> {
  
  if (!device) {
    throw new Error('WebGPU device not available');
  }
  
  // Create buffers - we need 5 buffers total:
  // 1. Uniform buffer (m, n, k, alpha)
  // 2. Input buffer A (matrix A data)
  // 3. Input buffer B (matrix B data) 
  // 4. Output buffer (result matrix C on GPU)
  // 5. Readback buffer (to copy results back to CPU)

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
  device.queue.writeBuffer(bBuffer, 0, array_b);
  device.queue.writeBuffer(aBuffer, 0, array_a);
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

  computePass.setPipeline(sgemmNaiveCoalescedPipeline);
  computePass.setBindGroup(0, bindGroup);
  
  // Calculate workgroup counts - adjust based on your compute.wgsl workgroup size
  // Typical workgroup size is 8x8 or 16x16 threads
  
  const workgroupsX = Math.ceil(n / workgroupSizeX);
  const workgroupsY = Math.ceil(m / workgroupSizeY);
  runtime.start_t = performance.now();
  computePass.dispatchWorkgroups(workgroupsX, workgroupsY, 1);
  computePass.end();

  // Copy result from GPU storage buffer to CPU-readable buffer
  commandEncoder.copyBufferToBuffer(resultBuffer, 0, readbackBuffer, 0, resultBufferSize);

  // Submit commands to GPU
  let commands = commandEncoder.finish();
  device.queue.submit([commands]);


  // Read back results from GPU to CPU
  await device.queue.onSubmittedWorkDone();

  runtime.end_t = performance.now();

  await readbackBuffer.mapAsync(GPUMapMode.READ);
  const resultArray = new Float32Array(readbackBuffer.getMappedRange().slice(0));
  //const mapped = readbackBuffer.getMappedRange();
  //const cPacked = new Float32Array(mapped.slice(0));
  readbackBuffer.unmap();


  // Clean up buffers (optional but good practice)
  uniformBuffer.destroy();
  aBuffer.destroy();
  bBuffer.destroy();
  resultBuffer.destroy();
  readbackBuffer.destroy();

  return resultArray;
}


async function sgemm3(m: number, n: number, k: number, alpha: number, array_a: Float32Array, array_b: Float32Array, runtime): Promise<Float32Array> {
  
  if (!device) {
    throw new Error('WebGPU device not available');
  }
  
  // Create buffers - we need 5 buffers total:
  // 1. Uniform buffer (m, n, k, alpha)
  // 2. Input buffer A (matrix A data)
  // 3. Input buffer B (matrix B data) 
  // 4. Output buffer (result matrix C on GPU)
  // 5. Readback buffer (to copy results back to CPU)

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
  device.queue.writeBuffer(bBuffer, 0, array_b);
  device.queue.writeBuffer(aBuffer, 0, array_a);
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

  computePass.setPipeline(sgemmBlockedPipeline);
  computePass.setBindGroup(0, bindGroup);
  
  // Calculate workgroup counts - adjust based on your compute.wgsl workgroup size
  // Typical workgroup size is 8x8 or 16x16 threads
  
  const workgroupsX = Math.ceil(n / workgroupSizeX);
  const workgroupsY = Math.ceil(m / workgroupSizeY);
  runtime.start_t = performance.now();
  computePass.dispatchWorkgroups(workgroupsX, workgroupsY, 1);
  computePass.end();

  // Copy result from GPU storage buffer to CPU-readable buffer
  commandEncoder.copyBufferToBuffer(resultBuffer, 0, readbackBuffer, 0, resultBufferSize);

  // Submit commands to GPU
  let commands = commandEncoder.finish();
  device.queue.submit([commands]);


  // Read back results from GPU to CPU
  await device.queue.onSubmittedWorkDone();

  runtime.end_t = performance.now();

  await readbackBuffer.mapAsync(GPUMapMode.READ);
  const resultArray = new Float32Array(readbackBuffer.getMappedRange().slice(0));
  //const mapped = readbackBuffer.getMappedRange();
  //const cPacked = new Float32Array(mapped.slice(0));
  readbackBuffer.unmap();


  // Clean up buffers (optional but good practice)
  uniformBuffer.destroy();
  aBuffer.destroy();
  bBuffer.destroy();
  resultBuffer.destroy();
  readbackBuffer.destroy();

  return resultArray;
}



async function sgemm4(m: number, n: number, k: number, alpha: number, array_a: Float32Array, array_b: Float32Array, runtime): Promise<Float32Array> {
  
  if (!device) {
    throw new Error('WebGPU device not available');
  }
  
  // Create buffers - we need 5 buffers total:
  // 1. Uniform buffer (m, n, k, alpha)
  // 2. Input buffer A (matrix A data)
  // 3. Input buffer B (matrix B data) 
  // 4. Output buffer (result matrix C on GPU)
  // 5. Readback buffer (to copy results back to CPU)

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
  device.queue.writeBuffer(bBuffer, 0, array_b);
  device.queue.writeBuffer(aBuffer, 0, array_a);
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

  computePass.setPipeline(sgemmBlockedPipeline);
  computePass.setBindGroup(0, bindGroup);
  
  // Calculate workgroup counts - adjust based on your compute.wgsl workgroup size
  // Typical workgroup size is 8x8 or 16x16 threads
  
  const workgroupsX = Math.ceil(n / workgroupSizeX);
  const workgroupsY = Math.ceil(m / workgroupSizeY);
  runtime.start_t = performance.now();
  computePass.dispatchWorkgroups(workgroupsX, workgroupsY, 1);
  computePass.end();

  // Copy result from GPU storage buffer to CPU-readable buffer
  commandEncoder.copyBufferToBuffer(resultBuffer, 0, readbackBuffer, 0, resultBufferSize);

  // Submit commands to GPU
  let commands = commandEncoder.finish();
  device.queue.submit([commands]);


  // Read back results from GPU to CPU
  await device.queue.onSubmittedWorkDone();

  runtime.end_t = performance.now();

  await readbackBuffer.mapAsync(GPUMapMode.READ);
  const resultArray = new Float32Array(readbackBuffer.getMappedRange().slice(0));
  //const mapped = readbackBuffer.getMappedRange();
  //const cPacked = new Float32Array(mapped.slice(0));
  readbackBuffer.unmap();


  // Clean up buffers (optional but good practice)
  uniformBuffer.destroy();
  aBuffer.destroy();
  bBuffer.destroy();
  resultBuffer.destroy();
  readbackBuffer.destroy();

  return resultArray;
}

