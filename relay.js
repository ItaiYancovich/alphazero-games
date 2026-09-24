// A private two-player "room" carried over public Nostr relays.
//
// Why relays and not a direct peer-to-peer connection: a direct WebRTC link
// needs a matchmaking server and, on strict networks, a TURN relay -- and
// school, office and some home networks block exactly those.  Nostr relays
// are ordinary secure WebSockets on port 443, run free by many independent
// operators, and forward short-lived ("ephemeral") messages to whoever is
// subscribed.  A turn-based game sends a handful of small messages per move,
// which is well within what they carry.
//
// Privacy: every message is compressed and then encrypted with AES-GCM under
// a key that exists only in the invite link's #fragment -- which browsers
// never send to any server.  The relays see a random-looking room tag and
// ciphertext.  Each browser signs with a throwaway key made for the session.
(function () {
  "use strict";

  const RELAYS = ["wss://relay.damus.io", "wss://nos.lol", "wss://relay.primal.net"];
  const KIND = 25050;                 // ephemeral: relayed live, never stored

  const b64u = {
    enc(bytes) {
      let s = "";
      for (let i = 0; i < bytes.length; i += 0x8000) {
        s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
      }
      return btoa(s).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
    },
    dec(text) {
      const s = atob(text.replace(/-/g, "+").replace(/_/g, "/"));
      const out = new Uint8Array(s.length);
      for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i);
      return out;
    },
  };

  async function pipe(bytes, stream) {
    const out = new Response(new Blob([bytes]).stream().pipeThrough(stream));
    return new Uint8Array(await out.arrayBuffer());
  }

  const hex = bytes => Array.from(bytes, b => b.toString(16).padStart(2, "0")).join("");

  class Room {
    /** `secret` is the invite: "<room id>.<key>", both base64url. */
    static create() {
      const id = b64u.enc(crypto.getRandomValues(new Uint8Array(9)));
      const key = b64u.enc(crypto.getRandomValues(new Uint8Array(16)));
      return new Room(`${id}.${key}`);
    }

    constructor(secret) {
      const [id, key] = String(secret).split(".");
      if (!id || !key) throw new Error("That invite link is incomplete.");
      this.secret = secret;
      this.id = id;
      this.rawKey = b64u.dec(key);
      this.sk = NostrTools.generateSecretKey();
      this.pk = NostrTools.getPublicKey(this.sk);
      this.sockets = new Map();       // url -> WebSocket
      this.seen = new Set();          // event ids already handled (relays overlap)
      this.handlers = [];
      this.statusHandlers = [];
      this.closed = false;
      this.since = Math.floor(Date.now() / 1000) - 10;
    }

    async open() {
      this.key = await crypto.subtle.importKey("raw", this.rawKey, "AES-GCM", false,
                                               ["encrypt", "decrypt"]);
      const digest = new Uint8Array(await crypto.subtle.digest(
        "SHA-256", new TextEncoder().encode("azgames-room:" + this.id)));
      this.tag = hex(digest).slice(0, 32);
      await Promise.any(RELAYS.map(url => this._connect(url)))
        .catch(() => { throw new Error("Could not reach any relay. Check the internet connection."); });
      return this;
    }

    on(fn) { this.handlers.push(fn); }
    onStatus(fn) { this.statusHandlers.push(fn); }
    get live() { return [...this.sockets.values()].some(ws => ws.readyState === 1); }

    _status() { for (const fn of this.statusHandlers) fn(this.live); }

    _connect(url, attempt = 0) {
      return new Promise((resolve, reject) => {
        if (this.closed) { reject(new Error("closed")); return; }
        let ws;
        try { ws = new WebSocket(url); } catch (err) { reject(err); return; }
        const timer = setTimeout(() => { try { ws.close(); } catch (e) { /* gone */ } reject(new Error("timeout")); }, 9000);
        ws.onopen = () => {
          clearTimeout(timer);
          this.sockets.set(url, ws);
          ws.send(JSON.stringify(["REQ", "azg", { kinds: [KIND], "#t": [this.tag], since: this.since }]));
          this._status();
          resolve(ws);
        };
        ws.onmessage = ev => this._incoming(ev.data);
        ws.onerror = () => { clearTimeout(timer); reject(new Error("socket error")); };
        ws.onclose = () => {
          clearTimeout(timer);
          if (this.sockets.get(url) === ws) this.sockets.delete(url);
          this._status();
          if (!this.closed) {
            // Reconnect with backoff; a relay that is down for good just stays out.
            const wait = Math.min(30000, 1500 * 2 ** Math.min(attempt, 5));
            setTimeout(() => this._connect(url, attempt + 1).catch(() => {}), wait);
          }
        };
      });
    }

    async _incoming(raw) {
      let msg;
      try { msg = JSON.parse(raw); } catch (e) { return; }
      if (msg[0] !== "EVENT" || !msg[2]) return;
      const ev = msg[2];
      if (ev.pubkey === this.pk || this.seen.has(ev.id)) return;
      this.seen.add(ev.id);
      if (this.seen.size > 2000) this.seen = new Set([...this.seen].slice(-500));
      if (!NostrTools.verifyEvent(ev)) return;
      try {
        const bytes = b64u.dec(ev.content);
        const plain = await crypto.subtle.decrypt({ name: "AES-GCM", iv: bytes.subarray(0, 12) },
                                                  this.key, bytes.subarray(12));
        const json = await pipe(new Uint8Array(plain), new DecompressionStream("deflate-raw"));
        const data = JSON.parse(new TextDecoder().decode(json));
        for (const fn of this.handlers) fn(data);
      } catch (err) {
        // Not for us (a different key), or damaged: ignore it.
      }
    }

    async send(data) {
      const packed = await pipe(new TextEncoder().encode(JSON.stringify(data)),
                                new CompressionStream("deflate-raw"));
      const iv = crypto.getRandomValues(new Uint8Array(12));
      const sealed = new Uint8Array(await crypto.subtle.encrypt({ name: "AES-GCM", iv }, this.key, packed));
      const body = new Uint8Array(12 + sealed.length);
      body.set(iv, 0);
      body.set(sealed, 12);
      const ev = NostrTools.finalizeEvent({
        kind: KIND, created_at: Math.floor(Date.now() / 1000),
        tags: [["t", this.tag]], content: b64u.enc(body),
      }, this.sk);
      const frame = JSON.stringify(["EVENT", ev]);
      let sent = 0;
      for (const ws of this.sockets.values()) {
        if (ws.readyState === 1) { ws.send(frame); sent++; }
      }
      if (!sent) throw new Error("not connected to any relay");
    }

    close() {
      this.closed = true;
      for (const ws of this.sockets.values()) { try { ws.close(); } catch (e) { /* gone */ } }
      this.sockets.clear();
    }
  }

  window.RelayRoom = Room;
})();
