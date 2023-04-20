// Get the button:
let btt_button = document.getElementById("back_to_top");

// When the user scrolls down 20px from the top of the document, show the button
window.onscroll = function() {scrollFunction()};

function scrollFunction() {
    if (document.body.scrollTop > 30 || document.documentElement.scrollTop > 30) {
        btt_button.style.display = "block";
    } else {
        btt_button.style.display = "none";
    }
}

// When the user clicks on the button, scroll to the top of the document
function backToTop() {
    document.body.scrollTop = 0; // For Safari
    document.documentElement.scrollTop = 0; // For Chrome, Firefox, IE and Opera
}