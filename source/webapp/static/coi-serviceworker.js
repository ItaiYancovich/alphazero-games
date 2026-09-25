// Cross-origin isolation for a static host that cannot set headers.
//
// The engine waits on shared memory for each network answer, and a page only
// gets SharedArrayBuffer when it is "cross-origin isolated" -- which takes two
// response headers.  GitHub Pages will not send them, so this service worker
// adds them to every response it serves.  The same file is loaded by the page
// as a plain script: there it registers itself, and reloads once when the
// worker takes control, since only the reload sees the new headers.
//
// The idea is the one popularised by gzuidhof/coi-serviceworker; this is a
// small independent version of it.

if (typeof window === "undefined") {
  // ------------------------------------------------------------ the worker
  const COEP = new URL(self.location).searchParams.get("coep") || "require-corp";
  self.addEventListener("install", () => self.skipWaiting());
  self.addEventListener("activate", event => event.waitUntil(self.clients.claim()));
  // The big runtimes never change under the same name; everything else (the
  // page, the engine, the networks) can change with any update, and GitHub
  // Pages tells browsers to keep a copy for ten minutes -- long enough that a
  // phone shows yesterday's site right after an update.  So those are always
  // revalidated: a cheap "not modified" when nothing changed, the new file
  // when something did.
  const STABLE = /\/(pyodide|ort)\//;
  const fresh = req => {
    const url = new URL(req.url);
    if (req.method !== "GET" || url.origin !== self.location.origin || STABLE.test(url.pathname)) {
      return fetch(req);
    }
    if (req.mode === "navigate") {
      return fetch(req.url, { cache: "no-cache", credentials: "same-origin" }).then(res =>
        res.redirected ? Response.redirect(res.url, 302) : res);
    }
    return fetch(new Request(req, { cache: "no-cache" }));
  };
  self.addEventListener("fetch", event => {
    const req = event.request;
    if (req.cache === "only-if-cached" && req.mode !== "same-origin") return;
    event.respondWith((async () => {
      const res = await fresh(req);
      if (res.type === "opaqueredirect" || res.status === 302) return res;
      if (res.status === 0) return res;             // opaque: pass through untouched
      const headers = new Headers(res.headers);
      headers.set("Cross-Origin-Embedder-Policy", COEP);
      headers.set("Cross-Origin-Opener-Policy", "same-origin");
      if (new URL(req.url).origin === self.location.origin) {
        headers.set("Cross-Origin-Resource-Policy", "same-origin");
      }
      return new Response(res.body, { status: res.status, statusText: res.statusText, headers });
    })());
  });
} else if (window.crossOriginIsolated) {
  try { sessionStorage.removeItem("coiReloaded"); } catch (e) { /* private mode */ }
} else if ("serviceWorker" in navigator && window.isSecureContext) {
  // ------------------------------------------------------------ the page
  // Chrome and Firefox accept `credentialless`, which lets the page keep
  // loading Google Fonts; Safari only knows `require-corp`.
  const ua = navigator.userAgent;
  const credentialless = /Chrome\/|Chromium\/|Firefox\//.test(ua);
  const script = document.currentScript.src.split("?")[0]
    + "?coep=" + (credentialless ? "credentialless" : "require-corp");
  const reloaded = sessionStorage.getItem("coiReloaded") === "1";
  navigator.serviceWorker.register(script).then(reg => {
    const reload = () => {
      if (reloaded) return;                         // never loop: one try per tab
      sessionStorage.setItem("coiReloaded", "1");
      location.reload();
    };
    if (navigator.serviceWorker.controller) reload();
    else navigator.serviceWorker.addEventListener("controllerchange", reload);
    if (reg.active && !navigator.serviceWorker.controller) reload();
  }).catch(err => console.warn("service worker:", err));
}
