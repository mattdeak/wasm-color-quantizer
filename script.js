document.addEventListener('DOMContentLoaded', () => {
    const imageUpload = document.getElementById('imageUpload');
    const colorCount = document.getElementById('colorCount');
    const processBtn = document.getElementById('processBtn');
    const downloadBtn = document.getElementById('downloadBtn');
    const imagePreview = document.getElementById('imagePreview');
    const previewImage = document.getElementById('previewImage');
    const dropText = document.getElementById('dropText');
    const processedImagePreview = document.getElementById('processedImagePreview');
    const processedImage = document.getElementById('processedImage');
    const processingIndicator = document.getElementById('processingIndicator');
    let uploadedImage = null;

    imageUpload.addEventListener('change', handleImageUpload);
    imagePreview.addEventListener('dragover', handleDragOver);
    imagePreview.addEventListener('drop', handleDrop);

    function handleImageUpload(e) {
        const file = e.target.files[0];
        if (file) {
            uploadedImage = file;
            displayImage(file);
            downloadBtn.disabled = true;
        }
    }

    function handleDragOver(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function handleDrop(e) {
        e.preventDefault();
        e.stopPropagation();
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            imageUpload.files = e.dataTransfer.files;
            handleImageUpload({ target: { files: [file] } });
        }
    }

    function displayImage(file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            previewImage.src = e.target.result;
            previewImage.style.display = 'block';
            dropText.style.display = 'none';
        }
        reader.readAsDataURL(file);
    }

    function predictOutputTime(totalPixels, maxColors) {
        return (
            6.7057e-06 * totalPixels +
                -4.6229e+00 * maxColors +
                8.4536e-03 * Math.log(totalPixels) +
                -8.1105e-01 * Math.log(maxColors) +
                3.1318e-07 * totalPixels * maxColors +
                -2.5181e-07 * totalPixels * Math.log(totalPixels) +
                -2.1015e-06 * totalPixels * Math.log(maxColors) +
                -8.5418e-02 * Math.pow(maxColors, 2) +
                -1.6574e-01 * maxColors * Math.log(totalPixels) +
                2.8262e+00 * maxColors * Math.log(maxColors) +
                -1.5364e-01 * Math.pow(Math.log(totalPixels), 2) +
                2.2803e+00 * Math.log(totalPixels) * Math.log(maxColors) +
                -8.1863e+00 * Math.pow(Math.log(maxColors), 2) +
                1.1316e+01  // Intercept
        ) * 3 // WASM slowdown factor (est)
    }

    function getBestSampleRate(totalPixels, maxColors) {
        // This function sucks and doesn't work
        let bestSampleRate = 1;

        for (let i = 1; i <= Math.min(totalPixels, 100); i++) {
            const sampledPixels = Math.ceil(totalPixels / i);
            const time = predictOutputTime(sampledPixels * 3, maxColors);
            console.log(`Sample rate for ${sampledPixels} pixels: ${time}ms`);
            if (time < 500) { // 1000 ms = 1 second
                bestSampleRate = i;
                break;
            }
        }
        return bestSampleRate;
    }

    processBtn.addEventListener('click', async () => {
        if (uploadedImage) {
            const before = performance.now();
            const img = new Image();
            img.onload = function() {
                const canvas = document.createElement('canvas');
                const ctx = canvas.getContext('2d');
                canvas.width = img.width;
                canvas.height = img.height;
                ctx.drawImage(img, 0, 0);
                const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height, {
                    colorSpace: 'srgb'
                });
                
                const numColors = parseInt(colorCount.value);

                console.log(`Image data length: ${imageData.data.length}`);
                const sampleRate = getBestSampleRate(imageData.data.length / 4, numColors);
                console.log(`Selected Sample rate: ${sampleRate}`);

                // data, max colors, sample rate, # channels in image data
                const processedData = reduce_colorspace(imageData.data, numColors, sampleRate, 4);
                const processedImageData = new ImageData(new Uint8ClampedArray(processedData), canvas.width, canvas.height, {
                    colorSpace: 'srgb'
                });
                ctx.putImageData(processedImageData, 0, 0);
                
                const processedImage = document.getElementById('processedImage');
                processedImage.src = canvas.toDataURL();
                processedImage.style.display = 'block';
                document.querySelector('#processedImagePreview p').style.display = 'none';
                downloadBtn.disabled = false;

                const after = performance.now();
                console.log(`Time taken: ${after - before}ms`);
            };
            img.src = URL.createObjectURL(uploadedImage);
        } else {
            alert('Please upload an image first');
        }
    });

    downloadBtn.addEventListener('click', () => {
        const processedImage = document.getElementById('processedImage');
        if (processedImage.src) {
            const a = document.createElement('a');
            a.href = processedImage.src;
            a.download = 'processed_image.png';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        }
    });
});