// Author: John Nemeth
// Sources: task material, ISPC documentation

export inline void vecblur(uniform float * uniform A[], uniform float * uniform B[], uniform int width, uniform int height) {

    foreach (c = 0 ... width, r = 0 ... height) {
      int tx = c + 1;
      int ty = r + 1;
      A[c][r] = (B[tx-1][ty-1] + B[tx][ty-1] + B[tx+1][ty-1] + B[tx-1][ty] + B[tx+1][ty] + B[tx-1][ty+1] + B[tx][ty+1] + B[tx+1][ty+1]) / 8.0;
    }
}
