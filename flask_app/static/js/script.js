// ================================================
// Credit Card Approval Prediction
// Form Validation
// ================================================

document.addEventListener("DOMContentLoaded", function () {

    const form = document.querySelector("form");

    form.addEventListener("submit", function (event) {

        const inputs = form.querySelectorAll("input[required]");

        for (let input of inputs) {

            if (input.value.trim() === "") {

                alert("Please fill all required fields.");

                input.focus();

                event.preventDefault();

                return;
            }
        }

    });

});