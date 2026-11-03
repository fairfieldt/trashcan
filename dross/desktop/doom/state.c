/* Acceptance-test observation, enabled only by an explicitly supplied channel. */
#include <stdint.h>
#include "doomstat.h"
#include "p_mobj.h"
int oscar_doom_state(int32_t out[5]) {
    if (gamestate != GS_LEVEL || players[consoleplayer].mo == NULL || leveltime < 5) return 0;
    out[0] = gametic; out[1] = leveltime;
    out[2] = players[consoleplayer].mo->x;
    out[3] = players[consoleplayer].mo->y;
    out[4] = players[consoleplayer].mo->angle;
    return 1;
}
