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
                const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                
                const numColors = parseInt(colorCount.value);

                const processedData = reduce_colorspace(canvas.width, canvas.height, imageData.data, numColors);
                
                const processedImageData = new ImageData(new Uint8ClampedArray(processedData), canvas.width, canvas.height);
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