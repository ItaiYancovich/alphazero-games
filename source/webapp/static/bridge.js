// The web build's side of the page: where `api()` goes instead of a server.
//
// Locally, every request goes to the engine worker -- the project's Python,
// running in this browser -- and back.  Online, one browser hosts: its engine
// is the game, and the friend's page sends its requests through an encrypted
// relay room (relay.js) to be answered there.  Either way the page itself cannot tell:
// `WEB_API(path, body)` resolves to exactly what the desktop server would have
// answered.
(function () {
  "use strict";

  const ls = {
    get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* private mode */ } },
  };

  // ------------------------------------------------------------ utilities
  const $ = id => document.getElementById(id);
  let nextId = 1;
  const pending = new Map();                     // request id -> {resolve, reject}

  function fail(message) {
    const box = $("boot-overlay");
    if (box) {
      box.classList.add("failed");
      $("boot-status").textContent = message;
      $("boot-bar").style.width = "100%";
    }
  }

  function setBoot(text, frac) {
    const s = $("boot-status"), b = $("boot-bar");
    if (s) s.textContent = text;
    if (b && frac != null) b.style.width = `${Math.round(frac * 100)}%`;
  }

  function hideBoot() {
    const box = $("boot-overlay");
    if (box) { box.classList.add("done"); setTimeout(() => box.remove(), 400); }
  }

  // An answer shaped like the desktop server's: 2xx and 409 are game states,
  // anything else is an error the page shows in its banner.
  function settle(status, body) {
    if (body && (status < 300 || status === 409)) return body;
    throw new Error((body && body.error) || `The game engine answered ${status}.`);
  }

  // ============================================================== engine
  let engine = null;                             // the engine worker, once started
  let engineReady = null;                        // promise: resolves when it answers

  function startEngine() {
    if (engineReady) return engineReady;
    engineReady = new Promise((resolve, reject) => {
      if (!self.crossOriginIsolated || typeof SharedArrayBuffer === "undefined") {
        reject(new Error("This browser cannot run the game engine here (it needs "
          + "cross-origin isolation). Try a recent Chrome, Edge, Firefox or Safari."));
        return;
      }
      const stages = { python: 0.15, numpy: 0.45, engine: 0.75 };
      const read = key => { try { return JSON.parse(ls.get(key) || "null"); } catch (e) { return null; } };
      fetch("models/models.json").then(r => r.json()).then(models => {
        // The engine and its pool of network workers (pool.js).  How many of
        // them to use is learned on the device; `azTuning` keeps it.
        engine = EnginePool.spawn(models, {
          extra: { profiles: read("azProfiles") || {}, tuning: read("azTuning") },
        }).engine;
        engine.onmessage = ({ data }) => {
          if (data.type === "progress") setBoot(data.detail + "…", stages[data.stage]);
          else if (data.type === "ready") { setBoot("Ready", 1); resolve(); }
          else if (data.type === "fatal") reject(new Error(data.message));
          else if (data.type === "error") console.warn("engine:", data.message);
          else if (data.type === "profiles") ls.set("azProfiles", JSON.stringify(data.files));
          else if (data.type === "tuning") ls.set("azTuning", JSON.stringify(data.tuning));
          else if (data.type === "reply") {
            const p = pending.get(data.id);
            if (p) { pending.delete(data.id); p.resolve(data); }
          }
        };
        engine.onerror = ev => reject(new Error(ev.message || "the game engine failed to start"));
      }).catch(reject);
    });
    return engineReady;
  }

  async function local(path, body) {
    await startEngine();
    const id = nextId++;
    const reply = await new Promise(resolve => {
      pending.set(id, { resolve });
      engine.postMessage({ type: "request", id, path, body });
    });
    return reply;                                 // {status, body}
  }


  // ============================================================== online
  // One browser hosts and its engine is the game; the other joins by link.
  // Both seats are humans as far as the engine knows; `remote_seats` says which
  // one is played from the other browser, and each side's copy of the state is
  // "personalised" so the page lets exactly the right person move.
  //
  // Messages (all through the encrypted room, see relay.js):
  //   guest -> host  join {cid}            until welcomed
  //   host  -> guest welcome {to, state}   and then state {state} on every change
  //   guest -> host  req {cid, id, path, body}
  //   host  -> guest res {to, id, status, body}
  //   both           ping {cid} every 10 s; bye {cid} on leaving
  const HEARTBEAT = 10000, QUIET = 35000;
  const online = {
    role: null,            // null | "host" | "guest"
    room: null,
    connected: false,      // a friend is (or was very recently) there
    guestSeat: 2,          // the engine seat the friend plays
    friend: null,          // host: the guest's client id
    lastSeen: 0,
    pushed: -1,            // host: last revision sent to the guest
    latest: null,          // guest: the most recent state from the host
    beat: null,
  };
  const cid = Math.random().toString(36).slice(2, 12);
  // Decided before the page's first request: a guest must never start an
  // engine of its own, and the page asks for /api/bots before the DOM is done.
  const JOIN = /#join=([\w-]+\.[\w-]+)/.exec(location.hash);
  if (JOIN) online.role = "guest";

  function seatsOf(st, role) {
    const remote = new Set(st.remote_seats || []);
    const all = st.seat_ids || [1, 2];
    return role === "guest" ? all.filter(s => remote.has(s)) : all.filter(s => !remote.has(s));
  }

  // The same state, told from one side of the table.
  function personalise(st, role) {
    if (!role || !st || !st.seat_ids) return st;
    const mine = new Set(seatsOf(st, role));
    st.your_turn = !!st.your_turn && mine.has(st.to_move);
    const who = seat => (mine.has(seat) ? "you" : "friend");
    for (const side of [...(st.seats || []), st.black, st.white]) {
      if (side && side.human) {
        side.name = who(side.seat);
        // To the page, "human" means "the person at this screen": the friend's
        // seat reads like an opponent's, so the status says whose turn it is.
        side.human = mine.has(side.seat);
      }
    }
    for (const m of st.moves || []) {
      if (m.by === "you" || m.by === "friend") m.by = who(m.player);
    }
    return st;
  }

  const clone = x => JSON.parse(JSON.stringify(x));
  const stateOf = async () => (await local("/api/state")).body;
  const withProblem = async (problem, status = 200) =>
    ({ status, body: Object.assign(await stateOf(), { problem }) });

  // ------------------------------------------------------------------ host
  // What either player may ask of the host's engine during an online game.
  async function hostHandle(path, body, fromGuest) {
    if (path === "/api/opponent") {
      return withProblem("Players are fixed during an online game — end it to play a bot.");
    }
    if (path === "/api/analysis" && body && body.enabled) {
      return withProblem("Live evaluation is off during online games — no hints for either side.");
    }
    if (path === "/api/user" && fromGuest) {
      return withProblem("Player profiles belong to the host's browser.");
    }
    if (path === "/api/move") {
      const st = await stateOf();
      if (!seatsOf(st, fromGuest ? "guest" : "host").includes(st.to_move)) {
        return withProblem(fromGuest ? "it is not your turn" : "it is your friend's turn", 409);
      }
    }
    if (path === "/api/new") return rematch(body || {});
    return local(path, body);
  }

  // A game between the two of you: every seat human, the friend's seat remote.
  let gamesMeta = null;
  async function rematch(body) {
    const st = await stateOf();
    const req = Object.assign({}, body, {
      game: body.game || st.game, rated: false,
      black: "human", white: "human",
      players: [{ choice: "human" }, { choice: "human" }],
    });
    // Online games are duels: Splendor is played two-handed.
    if (!gamesMeta) gamesMeta = (await local("/api/bots")).body.games;
    const game = gamesMeta.find(g => g.key === req.game);
    if (game && game.max_players > 2) req.size = "2";
    else if (!body.size && req.game === st.game) req.size = st.size;
    const reply = await local("/api/new", req);
    await local("/api/remote", { seats: [online.guestSeat] });
    const out = await stateOf();
    if (reply.body && reply.body.problem) out.problem = reply.body.problem;
    return { status: reply.status, body: out };
  }

  async function pushState(force) {
    if (online.role !== "host" || !online.friend || !online.room) return;
    const st = await stateOf();
    if (!force && st.revision === online.pushed) return;
    online.pushed = st.revision;
    online.room.send({ t: "state", to: online.friend, state: personalise(clone(st), "guest") })
      .catch(err => console.warn("send:", err));
  }

  async function hostStart() {
    await startEngine();
    if (online.room) return;
    online.role = "host";
    setOnline("Starting…");
    const room = RelayRoom.create();
    online.room = room;
    room.on(msg => hostMessage(msg).catch(err => console.warn("host:", err)));
    room.onStatus(live => { if (!live) setOnline("Reconnecting…"); else updateChip(); });
    try {
      await room.open();
    } catch (err) {
      online.room = null;
      online.role = null;
      setOnline("");
      throw err;
    }
    renderInvite(room.secret);
    setOnline("Waiting for your friend");
    startHeartbeat();
  }

  async function hostMessage(msg) {
    const room = online.room;
    if (!msg || !room) return;
    if (msg.t === "join") {
      if (online.friend && online.friend !== msg.cid && online.connected) {
        room.send({ t: "busy", to: msg.cid }).catch(() => {});
        return;
      }
      const first = online.friend !== msg.cid;
      online.friend = msg.cid;
      online.lastSeen = Date.now();
      online.connected = true;
      if (first) {
        // A fresh game between the two of you, in whatever is on the board.
        await rematch({});
        toast("Your friend joined. Good game!");
        closeDialog();
        refreshPage();
      }
      updateChip();
      const st = await stateOf();
      online.pushed = st.revision;
      room.send({ t: "welcome", to: msg.cid, state: personalise(clone(st), "guest") })
        .catch(err => console.warn("send:", err));
      return;
    }
    if (msg.cid !== online.friend) return;
    online.lastSeen = Date.now();
    if (!online.connected) { online.connected = true; updateChip(); }
    if (msg.t === "req") {
      let reply;
      try {
        reply = await hostHandle(msg.path, msg.body, true);
      } catch (err) {
        reply = { status: 500, body: { error: String(err.message || err) } };
      }
      const body = reply.body && reply.body.revision != null
        ? personalise(clone(reply.body), "guest") : reply.body;
      if (body && body.revision != null) online.pushed = body.revision;
      await room.send({ t: "res", to: msg.cid, id: msg.id, status: reply.status, body });
      refreshPage();                      // the host sees the friend's move at once
    } else if (msg.t === "bye") {
      lost("Your friend left the game.");
    }
  }

  // ----------------------------------------------------------------- guest
  let resolveGuest = null;
  const guestConnected = new Promise(r => { resolveGuest = r; });

  async function guestJoin(secret) {
    setBoot("Connecting to your friend's game…", 0.4);
    let room;
    try {
      room = new RelayRoom(secret);
      online.room = room;
      room.on(guestMessage);
      room.onStatus(live => { if (!live) setOnline("Reconnecting…"); else updateChip(); });
      await room.open();
    } catch (err) {
      fail(err.message);
      return;
    }
    setBoot("Waiting for your friend's game to answer…", 0.7);
    // Knock until the host answers: its page may still be starting up.
    let tries = 0;
    const knock = () => {
      if (online.connected || online.role !== "guest") return;
      if (++tries > 45) {
        fail("Your friend's game is not answering. Ask them to keep their page open "
             + "and send you a fresh link.");
        return;
      }
      room.send({ t: "join", cid }).catch(() => {});
      setTimeout(knock, 2000);
    };
    knock();
  }

  function guestMessage(msg) {
    if (!msg || (msg.to && msg.to !== cid)) return;
    online.lastSeen = Date.now();
    if (msg.t === "welcome") {
      online.latest = msg.state;
      if (!online.connected) {
        online.connected = true;
        updateChip();
        startHeartbeat();
        resolveGuest();
      }
      refreshPage();
    } else if (msg.t === "state") {
      if (!online.latest || msg.state.revision >= online.latest.revision) online.latest = msg.state;
      refreshPage();
    } else if (msg.t === "res") {
      if (msg.body && msg.body.revision != null
          && (!online.latest || msg.body.revision >= online.latest.revision)) {
        online.latest = msg.body;
      }
      const p = pending.get(msg.id);
      if (p) { pending.delete(msg.id); p.resolve({ status: msg.status, body: msg.body }); }
    } else if (msg.t === "busy") {
      fail("That game already has two players. Ask your friend for a new link.");
    } else if (msg.t === "bye") {
      lost("Your friend ended the online game.");
    } else if (msg.t === "ping" && !online.connected && online.latest) {
      online.connected = true;
      updateChip();
    }
  }

  async function remote(path, body) {
    await guestConnected;
    if (path === "/api/state" && body === undefined) return { status: 200, body: clone(online.latest) };
    if (path.startsWith("/api/training")) return { status: 200, body: {} };
    const id = nextId++;
    return new Promise((resolve, reject) => {
      pending.set(id, { resolve });
      online.room.send({ t: "req", cid, id, path, body }).catch(err => {
        pending.delete(id);
        reject(new Error("Not connected: " + err.message));
      });
      setTimeout(() => {
        if (pending.has(id)) {
          pending.delete(id);
          reject(new Error("Your friend's game did not answer. Is their page still open?"));
        }
      }, 20000);
    });
  }

  // ------------------------------------------------------------ both sides
  function startHeartbeat() {
    clearInterval(online.beat);
    online.beat = setInterval(() => {
      if (!online.room) return;
      online.room.send({ t: "ping", cid }).catch(() => {});
      if (online.connected && online.lastSeen && Date.now() - online.lastSeen > QUIET) {
        online.connected = false;
        setOnline(online.role === "guest" ? "Host not answering…" : "Friend not answering…");
      }
    }, HEARTBEAT);
  }

  function lost(message) {
    if (!online.role) return;
    toast(message);
    if (online.role === "guest") {
      online.connected = false;
      showDisconnected(message);
      return;
    }
    // The host keeps its board; the friend's seat goes back to being local.
    online.friend = null;
    online.connected = false;
    local("/api/remote", { seats: [] }).then(refreshPage);
    setOnline("Waiting for your friend");
  }

  function stopOnline(reload) {
    const room = online.room;
    if (room) {
      room.send({ t: "bye", cid }).catch(() => {}).finally(() => setTimeout(() => room.close(), 300));
    }
    clearInterval(online.beat);
    const wasGuest = online.role === "guest";
    Object.assign(online, { role: null, room: null, connected: false, friend: null, pushed: -1 });
    setOnline("");
    if (reload || wasGuest) {
      history.replaceState(null, "", location.pathname + location.search);
      location.reload();
    } else {
      local("/api/remote", { seats: [] }).then(refreshPage);
    }
  }

  // ----------------------------------------------------------- the page API
  let stateInFlight = null;
  window.WEB_API = async function (path, body) {
    // One poll at a time: a slow answer should not queue up a dozen more.
    if (path === "/api/state" && body === undefined && stateInFlight) return stateInFlight;
    const run = (async () => {
      if (online.role === "guest") {
        const reply = await remote(path, body);
        return settle(reply.status, reply.body);
      }
      const hosting = online.role === "host" && !!online.friend;
      const reply = hosting && body !== undefined ? await hostHandle(path, body, false)
                                                  : await local(path, body);
      const out = settle(reply.status, reply.body);
      if (hosting && out.revision != null) {
        pushState(false);
        return personalise(out, "host");
      }
      return out;
    })();
    if (path === "/api/state" && body === undefined) {
      stateInFlight = run;
      run.finally(() => { stateInFlight = null; }).catch(() => {});
    }
    return run;
  };

  // The page's own poll, run now rather than at its next tick.  `refresh` is a
  // top-level const of the page's script: reachable by name, not on window.
  function refreshPage() {
    try { refresh(); } catch (e) { /* page not started yet */ }
  }

  // ============================================================== chrome
  function setOnline(text) {
    document.body.classList.toggle("online", !!online.role);
    document.body.classList.toggle("online-guest", online.role === "guest");
    document.body.classList.toggle("online-host", online.role === "host");
    const chip = $("online-chip");
    if (!chip) return;
    chip.hidden = !online.role;
    $("online-chip-text").textContent = text || "Online";
    chip.classList.toggle("live", online.connected);
  }

  function updateChip() {
    if (!online.role) return setOnline("");
    if (online.connected) setOnline(online.role === "guest" ? "Playing your friend" : "Friend connected");
    else setOnline(online.role === "guest" ? "Connecting…" : "Waiting for your friend");
  }

  function renderInvite(secret) {
    $("invite-link").value = `${location.origin}${location.pathname}#join=${secret}`;
    $("invite-ready").hidden = false;
    $("invite-wait").hidden = true;
    $("invite-share").hidden = !navigator.share;
  }

  function openDialog() {
    const d = $("online-dialog");
    $("invite-error").hidden = true;
    const live = (online.role === "host" && online.friend) || online.role === "guest";
    $("invite-body").hidden = !!live;
    $("online-live").hidden = !live;
    if (live) {
      $("online-live-text").textContent = online.role === "guest"
        ? "You're playing your friend's game. Either of you can switch games with the "
          + "tabs at the top, and you both move to it together."
        : "You're connected to your friend. Switch games with the tabs at the top — "
          + "you both move to the new game together.";
    }
    if (typeof d.showModal === "function") { if (!d.open) d.showModal(); } else d.setAttribute("open", "");
    if (!online.role) hostStart().catch(err => showDialogError(err.message));
  }

  function closeDialog() {
    const d = $("online-dialog");
    if (d && d.open) d.close();
  }

  function showDialogError(message) {
    const box = $("invite-error");
    box.textContent = message;
    box.hidden = false;
    $("invite-wait").hidden = true;
  }

  function showDisconnected(message) {
    const box = $("boot-overlay-lost");
    if (box) { $("lost-text").textContent = message; box.hidden = false; }
  }

  let toastTimer = null;
  function toast(message) {
    const t = $("web-toast");
    if (!t) return;
    t.textContent = message;
    t.classList.add("on");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => t.classList.remove("on"), 4000);
  }

  function wire() {
    $("online-btn").addEventListener("click", openDialog);
    $("online-chip").addEventListener("click", openDialog);
    $("invite-close").addEventListener("click", closeDialog);
    $("invite-copy").addEventListener("click", async () => {
      const input = $("invite-link");
      try { await navigator.clipboard.writeText(input.value); }
      catch (e) { input.select(); document.execCommand("copy"); }
      toast("Link copied — send it to your friend");
    });
    $("invite-share").addEventListener("click", () => {
      navigator.share({ title: "Play me online", text: "Join my game:",
                        url: $("invite-link").value }).catch(() => {});
    });
    $("invite-seat").addEventListener("change", async () => {
      online.guestSeat = $("invite-seat").value === "first" ? 1 : 2;
      if (online.friend) { await rematch({}); pushState(true); refreshPage(); }
    });
    $("invite-stop").addEventListener("click", () => { closeDialog(); stopOnline(false); });
    $("live-leave").addEventListener("click", () => {
      closeDialog();
      stopOnline(online.role === "guest");
    });
    $("live-rematch").addEventListener("click", async () => {
      closeDialog();
      try { await window.WEB_API("/api/new", { players: [{}, {}] }); }
      catch (err) { toast(err.message); }
      refreshPage();
    });
    $("guest-leave").addEventListener("click", () => stopOnline(online.role === "guest"));
    $("lost-offline").addEventListener("click", () => stopOnline(true));
    window.addEventListener("pagehide", () => {
      if (online.room) online.room.send({ t: "bye", cid }).catch(() => {});
    });
  }

  // ============================================================== start
  function begin() {
    wire();
    if (online.role === "guest") {
      setOnline("Connecting…");
      guestJoin(JOIN[1]);
      guestConnected.then(hideBoot);
      return;
    }
    setBoot("Loading the game engine…", 0.05);
    startEngine().then(hideBoot).catch(err => fail(err.message));
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", begin);
  else begin();
})();
