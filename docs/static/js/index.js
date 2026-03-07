window.HELP_IMPROVE_VIDEOJS = false;

var INTERP_BASE = "https://homes.cs.washington.edu/~kpar/nerfies/interpolation/stacked";
var NUM_INTERP_FRAMES = 240;

var interp_images = [];
function preloadInterpolationImages() {
  for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path = INTERP_BASE + '/' + String(i).padStart(6, '0') + '.jpg';
    interp_images[i] = new Image();
    interp_images[i].src = path;
  }
}

function setInterpolationImage(i) {
  var image = interp_images[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#interpolation-image-wrapper').empty().append(image);
}


$(document).ready(function() {
    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");

    });

    var options = {
			slidesToScroll: 1,
			slidesToShow: 3,
			loop: true,
			infinite: true,
			autoplay: false,
			autoplaySpeed: 3000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    // Loop on each carousel initialized
    for(var i = 0; i < carousels.length; i++) {
    	// Add listener to  event
    	carousels[i].on('before:show', state => {
    		console.log(state);
    	});
    }

    // Access to bulmaCarousel instance of an element
    var element = document.querySelector('#my-element');
    if (element && element.bulmaCarousel) {
    	// bulmaCarousel instance is available as element.bulmaCarousel
    	element.bulmaCarousel.on('before-show', function(state) {
    		console.log(state);
    	});
    }

    /*var player = document.getElementById('interpolation-video');
    player.addEventListener('loadedmetadata', function() {
      $('#interpolation-slider').on('input', function(event) {
        console.log(this.value, player.duration);
        player.currentTime = player.duration / 100 * this.value;
      })
    }, false);*/
    preloadInterpolationImages();

    $('#interpolation-slider').on('input', function(event) {
      setInterpolationImage(this.value);
    });
    setInterpolationImage(0);
    $('#interpolation-slider').prop('max', NUM_INTERP_FRAMES - 1);

    bulmaSlider.attach();

})


// This is based on: http://thenewcode.com/364/Interactive-Before-and-After-Video-Comparison-in-HTML5-Canvas
// With additional modifications based on: https://jsfiddle.net/7sk5k4gp/13/

function playVids(videoId) {
  var videoMerge = document.getElementById(videoId + "Merge");
  var vid = document.getElementById(videoId);
  
  var position = 0.5;
  var vidWidth = vid.videoWidth/2;
  var vidHeight = vid.videoHeight;

  var mergeContext = videoMerge.getContext("2d");

  
  if (vid.readyState > 3) {
      vid.play();

      function trackLocation(e) {
          // Normalize to [0, 1]
          bcr = videoMerge.getBoundingClientRect();
          position = ((e.pageX - bcr.x) / bcr.width);
      }
      function trackLocationTouch(e) {
          // Normalize to [0, 1]
          bcr = videoMerge.getBoundingClientRect();
          position = ((e.touches[0].pageX - bcr.x) / bcr.width);
      }

      videoMerge.addEventListener("mousemove",  trackLocation, false); 
      videoMerge.addEventListener("touchstart", trackLocationTouch, false);
      videoMerge.addEventListener("touchmove",  trackLocationTouch, false);


      function drawLoop() {
          mergeContext.drawImage(vid, 0, 0, vidWidth, vidHeight, 0, 0, vidWidth, vidHeight);
          var colStart = (vidWidth * position).clamp(0.0, vidWidth);
          var colWidth = (vidWidth - (vidWidth * position)).clamp(0.0, vidWidth);
          mergeContext.drawImage(vid, colStart+vidWidth, 0, colWidth, vidHeight, colStart, 0, colWidth, vidHeight);
          requestAnimationFrame(drawLoop);

          
          var arrowLength = 0.07 * vidHeight;
          var arrowheadWidth = 0.020 * vidHeight;
          var arrowheadLength = 0.04 * vidHeight;
          var arrowPosY = vidHeight / 10;
          var arrowWidth = 0.007 * vidHeight;
          var currX = vidWidth * position;

          // Draw circle
          mergeContext.arc(currX, arrowPosY, arrowLength*0.7, 0, Math.PI * 2, false);
          mergeContext.fillStyle = "#FFD79340";
          mergeContext.fill()
          //mergeContext.strokeStyle = "#444444";
          //mergeContext.stroke()
          
          // Draw border
          mergeContext.beginPath();
          mergeContext.moveTo(vidWidth*position, 0);
          mergeContext.lineTo(vidWidth*position, vidHeight);
          mergeContext.closePath()
          mergeContext.strokeStyle = "#444444";
          mergeContext.lineWidth = 3;            
          mergeContext.stroke();

          // Draw arrow
          mergeContext.beginPath();
          mergeContext.moveTo(currX, arrowPosY - arrowWidth/2);
          
          // Move right until meeting arrow head
          mergeContext.lineTo(currX + arrowLength/2 - arrowheadLength/2, arrowPosY - arrowWidth/2);
          
          // Draw right arrow head
          mergeContext.lineTo(currX + arrowLength/2 - arrowheadLength/2, arrowPosY - arrowheadWidth/2);
          mergeContext.lineTo(currX + arrowLength/2, arrowPosY);
          mergeContext.lineTo(currX + arrowLength/2 - arrowheadLength/2, arrowPosY + arrowheadWidth/2);
          mergeContext.lineTo(currX + arrowLength/2 - arrowheadLength/2, arrowPosY + arrowWidth/2);

          // Go back to the left until meeting left arrow head
          mergeContext.lineTo(currX - arrowLength/2 + arrowheadLength/2, arrowPosY + arrowWidth/2);
          
          // Draw left arrow head
          mergeContext.lineTo(currX - arrowLength/2 + arrowheadLength/2, arrowPosY + arrowheadWidth/2);
          mergeContext.lineTo(currX - arrowLength/2, arrowPosY);
          mergeContext.lineTo(currX - arrowLength/2 + arrowheadLength/2, arrowPosY  - arrowheadWidth/2);
          mergeContext.lineTo(currX - arrowLength/2 + arrowheadLength/2, arrowPosY);
          
          mergeContext.lineTo(currX - arrowLength/2 + arrowheadLength/2, arrowPosY - arrowWidth/2);
          mergeContext.lineTo(currX, arrowPosY - arrowWidth/2);

          mergeContext.closePath();

          mergeContext.fillStyle = "#444444";
          mergeContext.fill();

          
          
      }
      requestAnimationFrame(drawLoop);
  } 
}

Number.prototype.clamp = function(min, max) {
return Math.min(Math.max(this, min), max);
};
  
  
function resizeAndPlay(element)
{
var cv = document.getElementById(element.id + "Merge");
cv.width = element.videoWidth/2;
cv.height = element.videoHeight;
element.play();
element.style.height = "0px";  // Hide video without stopping it
  
playVids(element.id);
}
