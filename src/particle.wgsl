////////////////////////////////////////////////////////////////////////////////
// Vertex shader
////////////////////////////////////////////////////////////////////////////////
struct CameraUniforms
{
  mvp: mat4x4<f32>, // view * projection matrix
  right : vec3<f32>, // camera.right and up are used to determine the direction the particle "sprites" face
  _pad0 : f32,
  up : vec3<f32>,
  _pad1 : f32,
}; 

struct VertexInput {
  @location(0) instPos : vec3<f32>,
  @location(1) instColor : vec4<f32>,
  @location(2) corner : vec2<f32>,
};

struct VertexOutput
{
    @builtin(position) pos : vec4<f32>,
    @location(0) color : vec4<f32>,
    @location(1) corner : vec2<f32>,
};

@group(0) @binding(0) var<uniform> cam : CameraUniforms;
@vertex
fn vs_main(in : VertexInput) -> VertexOutput {
  let size = 0.02;
  let worldPos = in.instPos + cam.right * (in.corner.x * size) + cam.up * (in.corner.y * size);

  var out: VertexOutput;
  out.pos = cam.mvp * vec4<f32>(worldPos, 1.0);
  out.color = in.instColor;
  out.corner = in.corner; 
  return out;

}

////////////////////////////////////////////////////////////////////////////////
// Fragment shader
////////////////////////////////////////////////////////////////////////////////
@fragment
fn fs_main(in : VertexOutput) -> @location(0) vec4f {
  var color = in.color;
  
  //var color = vec4f(1,1,1,1);
  color.a = color.a * max(1.0 - length(in.corner), 0.0);
  return color;
}

////////////////////////////////////////////////////////////////////////////////
// Simulation Compute shader
////////////////////////////////////////////////////////////////////////////////

struct SimulationParams {
  dt : f32,
  bounce : f32,
  _pad0 : vec2<f32>,

  count: u32,
  gravity: f32,
  _pad6: f32,
  _pad1: f32,
  
  boxMin : vec3<f32>,
  _pad2 : f32,

  boxMax : vec3<f32>,
  _pad3 : f32,

};

struct Particle {
  //pos life color vel and pad 1
  pos : vec3<f32>,
  life: f32,

  color: vec4<f32>,

  vel: vec3<f32>,
  _pad0: f32,
};


@group(0) @binding(0) var<uniform> sim : SimulationParams;
@group(0) @binding(1) var<storage, read_write> particles : array<Particle>;


@compute @workgroup_size(64) // test other workgroup sizes
fn simulate(@builtin(global_invocation_id) gid : vec3u) {
   let i = gid.x;
   if (i >= sim.count) {return;}

   var p = particles[i];
   p.vel = p.vel + vec3(0,0, sim.gravity * sim.dt); 
   p.pos = p.pos + p.vel * sim.dt;
   // collision handling
  if (p.pos.x < sim.boxMin.x) {
    p.pos.x = sim.boxMin.x;
    p.vel.x = abs(p.vel.x) * sim.bounce;
  } else if (p.pos.x > sim.boxMax.x) {
    p.pos.x = sim.boxMax.x;
    p.vel.x = -abs(p.vel.x) * sim.bounce;
  }

  if (p.pos.y < sim.boxMin.y) {
    p.pos.y = sim.boxMin.y;
    p.vel.y = abs(p.vel.y) * sim.bounce;
  } else if (p.pos.y > sim.boxMax.y) {
    p.pos.y = sim.boxMax.y;
    p.vel.y = -abs(p.vel.y) * sim.bounce;
  }

  if (p.pos.z < sim.boxMin.z) {
    p.pos.z = sim.boxMin.z;
    p.vel.z = abs(p.vel.z) * sim.bounce;
  } else if (p.pos.z > sim.boxMax.z) {
    p.pos.z = sim.boxMax.z;
    p.vel.z = -abs(p.vel.z) * sim.bounce;
  }

  particles[i] = p;
  }

  // Store the new particle value
  

