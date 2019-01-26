// Author: John Nemeth
// Sources: task material, ISPC documentation

export inline void vecblur(uniform float ** A, uniform float ** B, uniform int width, uniform int height) {

    foreach (c = 0 ... width, r = 0 ... height) {
      int tx = c + 1;
      int ty = r + 1;
      A[c][r] = (B[tX][tY] + B[tX-1][tY-1] + B[tX][tY-1] + B[tX+1][tY-1] + B[tX-1][tY] + B[tX+1][tY] + B[tX-1][tY+1] + B[tX][tY+1] + B[tX+1][tY+1]) / 9;
    }
}
