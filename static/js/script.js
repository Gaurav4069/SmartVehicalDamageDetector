// Function to display a preview of the selected image
function previewImage(event) {
    var fileInput = event.target;
    var file = fileInput.files[0];
    var reader = new FileReader();

    reader.onload = function() {
      var imgElement = document.getElementById("preview");
      if (imgElement) {
        imgElement.src = reader.result;
        imgElement.style.display = "block";
      }
    };

    if (file) {
      reader.readAsDataURL(file);
    }
}

// Add event listener to file input
var fileInput = document.getElementById("imageInput");
if (fileInput) {
  fileInput.addEventListener("change", previewImage);
}