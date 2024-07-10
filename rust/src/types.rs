#[derive(Clone, Copy, PartialEq, Debug)]
pub struct RGBAPixel(pub [f32; 3], pub u8);

impl RGBAPixel {
    pub fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        RGBAPixel([r as f32, g as f32, b as f32], a)
    }
}

impl From<RGBAPixel> for [u8; 4] {
    fn from(pixel: RGBAPixel) -> Self {
        [pixel.0[0] as u8, pixel.0[1] as u8, pixel.0[2] as u8, pixel.1]
    }
}

impl From<RGBAPixel> for [f32; 3] {
    fn from(pixel: RGBAPixel) -> Self {
        pixel.0
    }
}

impl From<&RGBAPixel> for [f32; 3] {
    fn from(pixel: &RGBAPixel) -> Self {
        pixel.0
    }
}

impl Eq for RGBAPixel {}


pub type Centroid = [f32; 3];