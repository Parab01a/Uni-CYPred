$(function () {
    // get current page url
    let url = location.href;
    // alert(url)
    // current page highlight status
    let status = false;
    // search every a tag in ul which id=navbar, and add highlight class
    $('#navbar a').each(function () {
        // match current page's url and add highlight
        if((url + '/').indexOf($(this).attr('href')) > -1 && $(this).attr('href') !== '') {
            // $(this).parent('li').addClass('highlight');
            $(this).addClass('highlight');
            status = true;
        }else {
            $(this).removeClass('highlight');
        }
    });
    // default: first is highlighted
    if(!status) {
        $('#navbar a').eq(0).addClass('highlight');
    }
})