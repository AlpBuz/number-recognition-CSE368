async function uploadImage(event){
    event.preventDefault();
    const image = document.getElementById('image').files[0];
    const formData = new FormData();
    if (image) {
        formData.append('image', image);


        const response = await fetch('/uploadImage', {
            method: 'POST',
            body: formData
        });

        if (response.ok){
            const data = await response.json();
            displayResults(data)
        }else{
            alert("something went wrong, try again")
        }
    }else{
        alert("no image was uploaded");
    }

}


function previewImage(event){
    const imageInput = event.target;
    const preview = document.getElementById('preview-image');
    const file = imageInput.files[0];

    if (file){
        const reader = new FileReader();
        reader.onload = function (e) {
            preview.src = e.target.result; // Set the preview image's src
        };
        reader.readAsDataURL(file);
    }
}

function displayResults(data){
    const numberPredicted = document.getElementById('result-text');
    const imageInput = document.getElementById('uploaded-image');

    if (data.filePath) {
        imageInput.src = data.filePath;
    }

    if (data.prediction){
        numberPredicted.textContent = data.prediction
    }else{
        numberPredicted.textContent = 'No prediction was made'
    }

}

function clearResults(){
    const numberPredicted = document.getElementById('result-text');
    const imageInput = document.getElementById('uploaded-image');

    imageInput.src = "";
    numberPredicted.textContent = "";
}