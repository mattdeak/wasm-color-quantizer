# WASM Color Quantizer
This is a simple tool I created to share a color reducer with a pixel artist friend. It's a quick but pretty good way to reduce the number of colors in an image using the K-means++ clustering algorithm.

<p align="center">
  <img src="https://github.com/mattdeak/wasm-color-quantizer/assets/17998873/158dc6ac-f899-4cd4-991d-61cf616890d8" alt="Demo Gif" />
</p>

## Access
You can access and use this tool [right here](https://mattdeak.github.io/wasm-color-quantizer/). It just runs in the browser.

## How it Works
This quantizer uses Rust compiled to WebAssembly (WASM) to perform the K-means calculation quickly and efficiently in the browser.
Will update soon with better sampling to handle very large N-color requests or humongous images (bigger than any reasonable image would be). Maybe gifs/video too.

I might eventually get around to splitting out the Rust package if I add enough functionality (other clustering methods, maybe), but feel free to clone this and rip it all out if you want.

Feel free to use this tool for whatever you like.

### WebGPU
If you have a browser that supports WebGPU, you can run the quantizer on GPU (under Advanced, select a GPU algorithm). This can be faster, particularly for big images, but Hamerly on CPU will be quicker on most images smaller than 1000x1000 if the color space.
