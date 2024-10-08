@group(0) @binding(0) var<storage, read> image: array<vec3<u32>>;
@group(0) @binding(1) var<storage, read> centers: array<vec3<f32>>;
@group(0) @binding(2) var<storage, read_write> assignments: array<u32>;

const WORKGROUP_SIZE = 256;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&image)) {
        return;
    }

    let pixel = image[idx];

    let fpixel = vec3<f32>(pixel);
    var min_dist = distance(fpixel, centers[0]);
    var min_center = 0u;

    for (var i = 1u; i < arrayLength(&centers); i++) {
        let dist = distance(fpixel, centers[i]);
        if (dist < min_dist) {
            min_dist = dist;
            min_center = i;
        }
    }

    assignments[idx] = min_center;
}

fn distance(a: vec3<f32>, b: vec3<f32>) -> f32 {
    let diff = a - b;
    return dot(diff, diff);
}