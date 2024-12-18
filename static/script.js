document.addEventListener('DOMContentLoaded', function () {
    const form = document.querySelector('form');
    const fileInput = document.querySelector('input[type="file"]');
    const submitButton = document.querySelector('input[type="submit"]');

    form.addEventListener('submit', function (event) {
        if (!fileInput.value) {
            alert("Please select a PDF file before submitting.");
            event.preventDefault(); // Prevent the form from submitting if no file is selected
        }
    });

    // Preview file name after selecting a file
    fileInput.addEventListener('change', function () {
        if (fileInput.files.length > 0) {
            const fileName = fileInput.files[0].name;
            alert(`You selected: ${fileName}`);
        }
    });
});
