document.getElementById("file").onchange = function() {
  document.getElementById("form").submit();
};

document.getElementById("upload-div").addEventListener("click", function(){
  document.getElementById("file").click();
});

if ( window.history.replaceState ) {
  window.history.replaceState( null, null, window.location.href );
}