// Two clients, one room tag, an ephemeral event from A to B through each relay.
import { generateSecretKey, finalizeEvent } from "nostr-tools";

const RELAYS = ["wss://relay.damus.io", "wss://nos.lol"];
const tag = "azg-test-" + Math.random().toString(36).slice(2);
const KIND = 25050;

function open(url) {
  return new Promise((ok, bad) => {
    const ws = new WebSocket(url);
    ws.onopen = () => ok(ws);
    ws.onerror = () => bad(new Error("cannot open " + url));
    setTimeout(() => bad(new Error("timeout " + url)), 8000);
  });
}

for (const url of RELAYS) {
  const t0 = performance.now();
  const [a, b] = await Promise.all([open(url), open(url)]);
  const got = new Promise(res => {
    b.onmessage = m => {
      const msg = JSON.parse(m.data);
      if (msg[0] === "EVENT") res(msg[2]);
      else if (msg[0] !== "EOSE") console.log("  b <-", JSON.stringify(msg).slice(0, 140));
    };
  });
  b.send(JSON.stringify(["REQ", "s1", { kinds: [KIND], "#t": [tag], since: Math.floor(Date.now() / 1000) - 5 }]));
  await new Promise(r => setTimeout(r, 400));
  a.onmessage = m => { const msg = JSON.parse(m.data); if (msg[0] === "OK") console.log("  a <- OK", msg[2], msg[3] || ""); };
  const ev = finalizeEvent({ kind: KIND, created_at: Math.floor(Date.now() / 1000),
                             tags: [["t", tag]], content: "x".repeat(20000) }, generateSecretKey());
  const sent = performance.now();
  a.send(JSON.stringify(["EVENT", ev]));
  const recv = await Promise.race([got, new Promise(r => setTimeout(() => r(null), 6000))]);
  console.log(url, recv ? `delivered in ${(performance.now() - sent).toFixed(0)} ms (20 KB)` : "NOT delivered",
              `connect ${(sent - t0).toFixed(0)} ms`);
  a.close(); b.close();
}
process.exit(0);
