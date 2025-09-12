// client/src/ws.ts
type Handler = (msg: any) => void;

export class AudioWS {
  private ws?: WebSocket;
  private url: string;
  private onMsg?: Handler;
  private onOpenCb?: () => void;
  private onCloseCb?: (ev: CloseEvent) => void;
  private onErrorCb?: (ev: Event) => void;

  constructor(url = `ws://${location.hostname}:8080/ws/audio`) {
    this.url = url;
  }

  onMessage(h: Handler) { this.onMsg = h; }
  onOpen(h: () => void) { this.onOpenCb = h; }
  onClose(h: (ev: CloseEvent) => void) { this.onCloseCb = h; }
  onError(h: (ev: Event) => void) { this.onErrorCb = h; }

  connect() {
    this.ws = new WebSocket(this.url);
    console.log("[WS] connecting to", this.url);
    this.ws.binaryType = "arraybuffer";
    this.ws.onopen = () => { console.log("[WS] open"); this.onOpenCb?.(); };
    this.ws.onclose = (e) => { console.log("[WS] close", e.code, e.reason); this.onCloseCb?.(e); };
    this.ws.onerror = (e) => { console.error("[WS] error", e); this.onErrorCb?.(e); };
    this.ws.onmessage = (e) => {
      if (typeof e.data === "string") {
        try {
          const obj = JSON.parse(e.data);
          console.log("[WS] msg", obj);
          this.onMsg?.(obj);
        } catch (err) {
          console.warn("[WS] bad json", e.data);
        }
      }
    };
  }  

  sendJSON(obj: any) {
    this.ws?.send(JSON.stringify(obj));
  }

  sendBinary(buf: ArrayBuffer) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(buf);
    }
  }
  

  start(sampleRate = 16000) {
    this.sendJSON({ type: "start", sample_rate: sampleRate });
  }

  bargeIn(ts: number) {
    this.sendJSON({ type: "barge_in", ts });
  }

  stop() {
    this.sendJSON({ type: "stop" });
  }

  close() {
    this.ws?.close();
  }

  get isOpen() {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}


