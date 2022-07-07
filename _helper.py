def find_center(x, y, w, h):
    cx = x + int(w/2)
    cy = y + int(h/2)
    return (cx, cy)

if __name__ == "__main__":
    print(find_center(10,10,100,25))