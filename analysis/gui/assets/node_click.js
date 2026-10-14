/* Make clicking a node of the 3D scene reliable.
 *
 * plotly.js only emits `plotly_click` for a 3D point when the cursor is within
 * 5 px of it while the button is held (hovering tolerates 10 px), which makes
 * the small node markers fiddly to click. This turns any mouse-up on the graph
 * that did not drag the camera into a click on the last hovered node, unless
 * plotly.js already emitted the click itself.
 */
(function () {
  var down = null;
  var nativeClickAt = 0;
  var lastHover = null;
  var hooked = null;

  function graphDiv() {
    var graph = document.getElementById("graph");
    return graph && graph.querySelector(".js-plotly-plot");
  }

  function hook(gd) {
    if (hooked === gd || !gd.on) return;
    hooked = gd;
    gd.on("plotly_click", function () {
      nativeClickAt = Date.now();
    });
    gd.on("plotly_hover", function (data) {
      lastHover = data && data.points && data.points[0];
    });
    gd.on("plotly_unhover", function () {
      // Picking pauses while the button is held: keep the hovered node then.
      if (!down) lastHover = null;
    });
  }

  function poll() {
    var gd = graphDiv();
    if (gd) hook(gd);
    setTimeout(poll, 1000);
  }
  poll();

  document.addEventListener(
    "mousedown",
    function (event) {
      var gd = graphDiv();
      down = gd && gd.contains(event.target) ? [event.clientX, event.clientY] : null;
    },
    true
  );

  document.addEventListener(
    "mouseup",
    function (event) {
      if (!down) return;
      var moved = Math.hypot(event.clientX - down[0], event.clientY - down[1]);
      down = null;
      var gd = graphDiv();
      var point = lastHover;
      if (!gd || !point || moved > 3 || Date.now() - nativeClickAt < 500) return;
      gd.emit("plotly_click", {
        points: [
          {
            curveNumber: point.curveNumber,
            pointNumber: point.pointNumber,
            x: point.x,
            y: point.y,
            z: point.z,
          },
        ],
      });
    },
    true
  );
})();
