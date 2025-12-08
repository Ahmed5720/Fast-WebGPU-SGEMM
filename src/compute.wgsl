const WorkGroup_X = 16;
const WorkGroup_Y = 16;
struct Uniforms {
    M: u32,
    N: u32,
    K: u32,
    alpha: u32,
};
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;





// Naive GEMM - each thread computes one output element
@compute @workgroup_size(WorkGroup_X, WorkGroup_Y,1)
fn naive(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let col = gid.y;
    let M = uniforms.M;
    let N = uniforms.N;
    let K = uniforms.K;
    if (row >= M || col >= N) { return; }

    let alpha = f32(uniforms.alpha);
    var sum: f32 = 0.0;
    // Simple dot product
    for (var k: u32 = 0u; k < K; k = k + 1u) {
        let a = A[row * K + k];
        let b = B[k * N + col];
        sum = sum + a * b;
    }
    C[row * N + col] = alpha * sum;
}


@compute @workgroup_size(WorkGroup_X, WorkGroup_Y,1)
fn naive_mem_coalesced(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;
    let M = uniforms.M;
    let N = uniforms.N;
    let K = uniforms.K;
    if (row >= M || col >= N) { return; }

    let alpha = f32(uniforms.alpha);
    var sum: f32 = 0.0;
    // Simple dot product
    for (var k: u32 = 0u; k < K; k = k + 1u) {
        let a = A[row * K + k];
        let b = B[k * N + col];
        sum = sum + a * b;
    }
    C[row * N + col] = alpha * sum;
}




// note that because WEBGPU does not allow work group synchronization in shaders with diverging control flow, we cant use bounds checking
// so we restrict ourselves to m,n,k dims that are stricly multiples of TILE
// // The block size must match the workgroup size (16)
const TILE: u32 = 16u   ;


var<workgroup> shA: array<array<f32, TILE>, TILE>;
var<workgroup> shB: array<array<f32, TILE>, TILE>;

@compute @workgroup_size(TILE, TILE, 1)
fn block_tiling(
  @builtin(workgroup_id)         wg_id: vec3<u32>,
  @builtin(local_invocation_id)  lid:   vec3<u32>,
  @builtin(global_invocation_id) gid:   vec3<u32>,
) {
  // CUDA mapping:
  // blockIdx  -> wg_id
  // threadIdx -> lid
  // i,j       -> gid.y, gid.x (since gid = wg_id*TILE + lid)

  let ty = lid.y;
  let tx = lid.x;

  let i = gid.y; // row in C (0..M-1)
  let j = gid.x; // col in C (0..N-1)

  var acc: f32 = 0.0;

  // number of tiles along K dimension
  let numPhases: u32 = uniforms.K / TILE;

  for (var p: u32 = 0u; p < numPhases; p = p + 1u) {
    // Load one TILE x TILE block of A and B into workgroup memory
    // A[i, p*TILE + tx]
    let aCol: u32 = p * TILE + tx;
    shA[ty][tx] = A[i * uniforms.K + aCol];

    // B[p*TILE + ty, j]
    let bRow: u32 = p * TILE + ty;
    shB[ty][tx] = B[bRow * uniforms.N + j];

    workgroupBarrier();

    // Dot product for this tile
    for (var k: u32 = 0u; k < TILE; k = k + 1u) {
      acc = acc + shA[ty][k] * shB[k][tx];
    }

    workgroupBarrier(); 
  }

  // Store C[i, j]
  C[i * uniforms.N + j] = acc;
}


var<workgroup> shvA: array<array<vec4<f32>, TILE / 4>, TILE>;
var<workgroup> shvB: array<array<vec4<f32>, TILE / 4>, TILE>;

@compute @workgroup_size(TILE / 4, TILE, 1)  // Adjust workgroup size
fn block_tiling_vectorized(
  @builtin(workgroup_id)         wg_id: vec3<u32>,
  @builtin(local_invocation_id)  lid:   vec3<u32>,
) {
  let ty = lid.y;
  let tx_vec = lid.x;  // Now each thread handles vec4
  
  let tile_i = wg_id.y * TILE + ty;
  let tile_j = wg_id.x * TILE + tx_vec * 4;
  
  var acc: array<vec4<f32>, 4> = array<vec4<f32>, 4>(
    vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0)
  );
  
  let numPhases: u32 = uniforms.K / TILE;
  
  for (var p: u32 = 0u; p < numPhases; p = p + 1u) {
    // Vectorized load of A
    let aCol = p * TILE + tx_vec * 4;
    shvA[ty][tx_vec] = vec4<f32>(
      A[tile_i * uniforms.K + aCol],
      A[tile_i * uniforms.K + aCol + 1],
      A[tile_i * uniforms.K + aCol + 2],
      A[tile_i * uniforms.K + aCol + 3]
    );
    
    // Vectorized load of B - each thread loads vec4 from B
    let bRow = p * TILE + ty;
    shvB[ty][tx_vec] = vec4<f32>(
      B[bRow * uniforms.N + tile_j],
      B[bRow * uniforms.N + tile_j + 1],
      B[bRow * uniforms.N + tile_j + 2],
      B[bRow * uniforms.N + tile_j + 3]
    );
    
    workgroupBarrier();
    
    // Compute 4x4 block matrix multiplication
    for (var k: u32 = 0u; k < TILE / 4; k = k + 1u) {
      let aVec = shvA[ty][k];
      let bVec = shvB[k][tx_vec];
      
      // Unroll for better performance
      acc[0] = fma(vec4<f32>(aVec.x), bVec, acc[0]);
      acc[1] = fma(vec4<f32>(aVec.y), bVec, acc[1]);
      acc[2] = fma(vec4<f32>(aVec.z), bVec, acc[2]);
      acc[3] = fma(vec4<f32>(aVec.w), bVec, acc[3]);
    }
    
    workgroupBarrier();
  }
  
  // Store 4 results per thread
  for (var comp: u32 = 0u; comp < 4u; comp = comp + 1u) {
    if (tile_j + comp < uniforms.N) {
      let idx = tile_i * uniforms.N + tile_j + comp;
      C[idx] = acc[comp].x + acc[comp].y + acc[comp].z + acc[comp].w;
    }
  }
}




const BLOCKSIZE: u32 = 16;
const TILE_M: u32 = 8;  // Tile size in M dimension
const TILE_N: u32 = 8;  // Tile size in N dimension

@compute @workgroup_size(BLOCKSIZE, BLOCKSIZE)
fn tiled(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y * TILE_M;
    let col = global_id.x * TILE_N;

    // initialize the array with all 0s
    var sums: array<array<f32, TILE_N>, TILE_M>;
    for (var i = 0u; i < TILE_M; i++) {
        for (var j = 0u; j < TILE_N; j++) {
            sums[i][j] = 0.0;
        }
    }

    // Compute the 2D tile
    for (var k = 0u; k < uniforms.K; k++) {
        // for each row
        for (var i = 0u; i < TILE_M; i++) {
            let a_element = A[(row + i) * uniforms.K + k];
            // calculate the dot product
            for (var j = 0u; j < TILE_N; j++) {
                let b_element = B[k * uniforms.N + (col + j)];
                sums[i][j] += a_element * b_element;
            }
        }
    }

    // Write results
    for (var i = 0u; i < TILE_M; i++) {
        for (var j = 0u; j < TILE_N; j++) {
            let output_row = row + i;
            let output_col = col + j;
            if (output_row < uniforms.M && output_col < uniforms.N) {
                C[output_row * uniforms.N + output_col] = sums[i][j];
            }
        }
    }
}
