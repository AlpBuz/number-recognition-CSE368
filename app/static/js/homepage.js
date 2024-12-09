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
        const data = await response.json();

        if (response.ok){

        }else{
            alert("something went wrong, try again")
        }
    }else{
        alert("no image was uploaded");
    }

}




function clearResults(){
    var uploaedImage = document.getElementById("uploaded-image");

    uploadImage.src = "";
}