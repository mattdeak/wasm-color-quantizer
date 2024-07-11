# K-Means Color Quantizer
This is a simple tool I created to share a color reducer with a pixel artist friend. It's a quick but pretty good way to reduce the number of colors in an image using the K-means++ clustering algorithm.

## Access

You can access and use this tool [right here](https://mattdeak.github.io/wasm-color-quantizer/). It just runs in the browser.

## How it Works

This color reducer uses Rust compiled to WebAssembly (WASM) to perform the K-means calculation quickly and efficiently in the browser.
Will update soon with better sampling to handle very large N-color requests or humongous images (bigger than any reasonable image would be).

Feel free to use this tool for whatever you like.
