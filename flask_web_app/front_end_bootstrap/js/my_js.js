/**
 * Created by shivendra on 10/04/18.
 */

$("#random").click(function(e) {
    e.preventDefault();
    $("#main_image").attr('src', "loading.png");
    $.ajax({
        type: "GET",
        url: "http://127.0.0.1:5000/random",
        // data: {
        //     id: $(this).val(), // < note use of 'this' here
        //     access_token: $("#access_token").val()
        // },
        success: function(result) {
            console.log(result);
            console.log('ok');
            $("#main_image").attr('src', "generated_images_for_logo_brewer.png"+"?"+(new Date()).getTime());
        },
        error: function(result) {
            console.log("Something went wrong");
            // $("#main_image").attr('src', "generated_images_for_logo_brewer.png"+"?"+(new Date()).getTime());
        }
    });
});


$(".my_color").click(function (e) {
    var ButtonText = $(this).text().toLowerCase();
    console.log(ButtonText);

    e.preventDefault();
    $("#main_image").attr('src', "loading.png");
    $.ajax({
        type: "POST",
        url: "http://127.0.0.1:5000/color",
        data: {
            color : ButtonText
        },
        success: function(result) {
            console.log('ok');
            $("#main_image").attr('src', ButtonText + ".png"+"?"+(new Date()).getTime());
        },
        error: function(result) {
            console.log("Something went wrong");
            // $("#main_image").attr('src', "generated_images_for_logo_brewer.png"+"?"+(new Date()).getTime());
        }
    });
});
