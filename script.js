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
    const algorithm = document.getElementById('algorithm');
    const maxIterations = document.getElementById('maxIterations');
    const tolerance = document.getElementById('tolerance');
    const seed = document.getElementById('seed');
    const algorithmWarning = document.getElementById('algorithmWarning');

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

    function getBestSampleRate(totalPixels, maxColors) {
        // TODO: Implement a better sample rate calculation
        return 1
    }

    processBtn.addEventListener('click', async () => {
        if (uploadedImage) {
            const before = performance.now();
            const img = new Image();
            img.onload = async function() {
                const canvas = document.createElement('canvas');
                const ctx = canvas.getContext('2d');
                canvas.width = img.width;
                canvas.height = img.height;
                ctx.drawImage(img, 0, 0);
                const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height, {
                    colorSpace: 'srgb'
                });
                
                const numColors = parseInt(colorCount.value);
                const selectedAlgorithm = algorithm.value;
                const maxIter = parseInt(maxIterations.value);
                const tol = parseFloat(tolerance.value);
                const seedValue = seed.value ? BigInt(seed.value) : undefined;

                console.log(`Image data length: ${imageData.data.length}`);
                const sampleRate = getBestSampleRate(imageData.data.length / 4, numColors);
                console.log(`Selected Sample rate: ${sampleRate}`);

                try {
                    let cruncherBuilder = new ColorCruncher(numColors, sampleRate)
                        .withAlgorithm(selectedAlgorithm)
                        .withMaxIterations(maxIter)
                        .withTolerance(tol);
                    
                    if (seedValue !== undefined) {
                        cruncherBuilder = cruncherBuilder.withSeed(seedValue);
                    }
                    
                    const cruncher = await cruncherBuilder.build();
                    const processedData = await cruncher.quantizeImage(imageData.data);
                    
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
                } catch (error) {
                    console.error("Error processing image:", error);
                    alert("An error occurred while processing the image. Please try again.");
                }
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

    algorithm.addEventListener('change', function() {
        algorithmWarning.style.display = this.value === 'lloyd-all-gpu' ? 'block' : 'none';
    });
});