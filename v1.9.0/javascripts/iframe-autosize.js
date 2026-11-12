// Resize Plotly iframe embeds to match their content height.
//
// Each themed-HTML figure produced by `plots/brahe_theme.save_themed_html`
// posts a `{type: 'brahe-fig-height', height: <pixels>}` message after
// Plotly finishes drawing. The listener below finds the iframe whose
// contentWindow sent the message and sets its height to the reported
// content height, so the iframe grows or shrinks to fit any figure size
// without each docs embed having to declare a CSS class up front.
//
// Why postMessage rather than directly setting iframe.height from the
// child: the iframe lives in the docs origin; the figure HTML is loaded
// from the same origin so a direct DOM reach is technically allowed,
// but postMessage keeps the contract one-way and survives a future
// move to a CDN-hosted figure directory.
(function () {
  function applyHeight(iframe, height) {
    if (!iframe || !Number.isFinite(height) || height <= 0) {
      return;
    }
    iframe.style.height = Math.round(height) + "px";
  }

  window.addEventListener("message", function (event) {
    var data = event && event.data;
    if (!data || data.type !== "brahe-fig-height") {
      return;
    }
    var iframes = document.querySelectorAll(".plotly-embed iframe");
    for (var i = 0; i < iframes.length; i++) {
      if (iframes[i].contentWindow === event.source) {
        applyHeight(iframes[i], data.height);
        return;
      }
    }
  });
})();
