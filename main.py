import psutil

def show_network_connections():
    connections = psutil.net_connections()
    for conn in connections:
        laddr = f"{conn.laddr.ip}:{conn.laddr.port}" if conn.laddr else ""
        raddr = f"{conn.raddr.ip}:{conn.raddr.port}" if conn.raddr else ""
        print(f"Proto: {conn.type}, Local: {laddr}, Remote: {raddr}, Status: {conn.status}, PID: {conn.pid}")

if __name__ == "__main__":
    show_network_connections()
    