/**
 * MaterialForge API & Socket Module
 * logic for backend communication.
 */
export class LabAPI {
  constructor(baseUrl) {
    this.baseUrl = baseUrl;
    this.socket = null;
    this.onMessageHandlers = [];
  }

  connect(path) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}${path}`;
    
    return new Promise((resolve, reject) => {
      this.socket = new WebSocket(wsUrl);
      this.socket.onopen = () => {
        console.log('[API] Connected to lab engine');
        resolve();
      };
      this.socket.onerror = (err) => reject(err);
      this.socket.onmessage = (msg) => {
        const data = JSON.parse(msg.data);
        this.onMessageHandlers.forEach(h => h(data));
      };
    });
  }

  send(type, data) {
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify({ type, data }));
    }
  }

  onMessage(handler) {
    this.onMessageHandlers.push(handler);
  }
}
