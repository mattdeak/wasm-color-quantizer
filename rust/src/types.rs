#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct RGBAPixel {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl RGBAPixel {
    pub fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        RGBAPixel { r, g, b, a }
    }
}

impl From<RGBAPixel> for [f32; 3] {
    fn from(pixel: RGBAPixel) -> Self {
        [pixel.r as f32, pixel.g as f32, pixel.b as f32]
    }
}

impl From<&RGBAPixel> for [f32; 3] {
    fn from(pixel: &RGBAPixel) -> Self {
        [pixel.r as f32, pixel.g as f32, pixel.b as f32]
    }
}

pub type Centroid = [f32; 3];