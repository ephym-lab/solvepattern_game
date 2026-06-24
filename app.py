import cv2, math, random, time, json, os
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_py
from mediapipe.tasks.python import vision as mp_vis

#CONFIG
PUZZLE_W, PUZZLE_H = 300, 300
SCORES_FILE = os.path.join(os.path.dirname(__file__), 'highscores.json')
DIFFICULTIES = {
    '1': ('Easy',   3, 3, 120),
    '2': ('Medium', 4, 4,  90),
    '3': ('Hard',   5, 5,  60),
}
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
    (13,17),(0,17),(17,18),(18,19),(19,20),
]
FINGERTIP_IDS = [4, 8, 12, 16, 20]
FINGER_PIP_IDS = [3, 6, 10, 14, 18]

# ── PARTICLES
class Particle:
    __slots__ = ['x','y','vx','vy','color','life','max_life','size','grav']
    def __init__(self, x, y, vx, vy, color, life, size=4, grav=0.25):
        self.x,self.y,self.vx,self.vy = float(x),float(y),vx,vy
        self.color,self.life,self.max_life,self.size,self.grav = color,life,life,size,grav
    def update(self):
        self.x+=self.vx; self.y+=self.vy; self.vy+=self.grav; self.vx*=0.96; self.life-=1
    def draw(self, frame):
        if self.life<=0: return
        a = self.life/self.max_life
        r = max(1, int(self.size*a))
        cv2.circle(frame,(int(self.x),int(self.y)),r,tuple(int(c*a) for c in self.color),cv2.FILLED)

def emit_burst(particles, cx, cy, count=40):
    COLS = [(0,255,200),(0,200,255),(255,200,0),(200,255,0),(255,120,200),(255,255,255)]
    for _ in range(count):
        a = random.uniform(0, 2*math.pi); spd = random.uniform(2,10)
        particles.append(Particle(cx,cy,math.cos(a)*spd,math.sin(a)*spd,
                                  random.choice(COLS),random.randint(20,45),random.randint(3,8)))

def emit_trail(particles, x, y):
    if random.random() < 0.55:
        c = random.choice([(80,180,255),(120,220,255),(200,255,255),(255,200,80)])
        particles.append(Particle(x+random.uniform(-5,5),y+random.uniform(-5,5),
                                  random.uniform(-0.8,0.8),random.uniform(-1.5,0.2),
                                  c,random.randint(8,18),random.randint(2,5),grav=0.04))

# PUZZLE PIECE  
class PuzzlePiece:
    def __init__(self, img, correct_pos, start_pos, size):
        self.image = img
        self.correct_pos = correct_pos
        self.x,self.y = float(start_pos[0]),float(start_pos[1])
        self.home_x,self.home_y = self.x,self.y
        self.size = size
        self.is_dragging = self.is_placed = False
        self.flash = 0          # >0 red penalty flash
        self.snap_glow = 0      # >0 green glow after snap

    def hovered(self, fx, fy):
        return self.x<=fx<=self.x+self.size[0] and self.y<=fy<=self.y+self.size[1]

# HIGH SCORES
def load_scores():
    try:
        if os.path.exists(SCORES_FILE):
            return json.load(open(SCORES_FILE))
    except Exception: pass
    return {n: [] for n,*_ in DIFFICULTIES.values()}

def save_score(scores, name, elapsed):
    scores.setdefault(name,[]).append(round(elapsed,2))
    scores[name] = sorted(scores[name])[:5]
    json.dump(scores, open(SCORES_FILE,'w'), indent=2)

# HELPERS    
def txt(frame, text, pos, scale, color, thick=2, font=cv2.FONT_HERSHEY_DUPLEX):
    x,y = pos
    cv2.putText(frame,text,(x+2,y+2),font,scale,(0,0,0),thick+2)
    cv2.putText(frame,text,pos,font,scale,color,thick)

def make_placeholder():
    img = np.zeros((PUZZLE_H,PUZZLE_W,3),dtype=np.uint8)
    for r in range(3):
        for c in range(3):
            col = [(200,80,80),(80,200,80),(80,80,200),
                   (200,200,80),(80,200,200),(200,80,200),
                   (160,120,60),(60,160,120),(120,60,160)][r*3+c]
            img[r*100:(r+1)*100, c*100:(c+1)*100] = col
            cv2.putText(img,str(r*3+c+1),(c*100+32,r*100+65),
                        cv2.FONT_HERSHEY_DUPLEX,1.4,(255,255,255),2)
    return img

def is_palm_open(lms, w, h):
    extended = sum(1 for t,p in zip(FINGERTIP_IDS,FINGER_PIP_IDS)
                   if lms[t].y < lms[p].y - 0.05)
    return extended >= 4

def build_pieces(src, cols, rows, pw, ph, tx, ty, fw, fh):
    spawn_mx = max(10, tx - pw - 20)
    pos = [(random.randint(10, spawn_mx), random.randint(60, max(65,fh-ph-10)))
           for _ in range(rows*cols)]
    random.shuffle(pos)
    out, i = [], 0
    for r in range(rows):
        for c in range(cols):
            seg = src[r*ph:(r+1)*ph, c*pw:(c+1)*pw].copy()
            out.append(PuzzlePiece(seg,(c,r),pos[i],(pw,ph))); i+=1
    return out

#MEDIAPIPE
detector = mp_vis.HandLandmarker.create_from_options(
    mp_vis.HandLandmarkerOptions(
        base_options=mp_py.BaseOptions(model_asset_path='hand_landmarker.task'),
        num_hands=2, min_hand_detection_confidence=0.65, min_tracking_confidence=0.65))

# WEBCAM + CONSTANTS
cap = cv2.VideoCapture(0)
if not cap.isOpened(): raise RuntimeError("Cannot open webcam.")
ret, _p = cap.read()
FH, FW = (_p.shape[:2] if ret else (480,640))
TX, TY = FW - PUZZLE_W - 20, 20

raw = cv2.imread('puzzle_source.png')
SRC_BASE = cv2.resize(raw if raw is not None else make_placeholder(),(PUZZLE_W,PUZZLE_H))
THUMB = cv2.resize(SRC_BASE,(90,90))

#GAME STATE
scores = load_scores()
state = 'menu'
diff_name,cols,rows,time_limit = DIFFICULTIES['1']
pw = ph = 100
pieces, particles = [], []
sel = {}
t_start = 0.0
G = dict(placed_count=0, streak=0, win_flash=0, penalty_flash=0)
palm_timers = {}
shuffle_cd = 0
fail_count = 0

def start_game(key):
    global state,diff_name,cols,rows,time_limit,pw,ph
    global pieces,particles,sel,t_start,palm_timers,shuffle_cd
    diff_name,cols,rows,time_limit = DIFFICULTIES[key]
    pw,ph = PUZZLE_W//cols, PUZZLE_H//rows
    src = cv2.resize(SRC_BASE,(PUZZLE_W,PUZZLE_H))
    pieces = build_pieces(src,cols,rows,pw,ph,TX,TY,FW,FH)
    particles,sel = [],{}
    t_start = time.time()
    G['placed_count']=G['streak']=G['win_flash']=G['penalty_flash']=0
    shuffle_cd=0; palm_timers={}
    state='playing'

def shuffle_pieces():
    spawn_mx = max(10, TX - pw - 20)
    for p in pieces:
        if not p.is_placed:
            p.x=float(random.randint(10,spawn_mx))
            p.y=float(random.randint(60,max(65,FH-ph-10)))
            p.home_x,p.home_y = p.x,p.y

# MAIN LOOP
try:
    while cap.isOpened():
        ok,frame = cap.read()
        if not ok:
            fail_count+=1
            if fail_count>=30: print("Camera dead."); break
            continue
        fail_count=0
        frame = cv2.flip(frame,1)
    
        # MENU
        if state == 'menu':
            ov = frame.copy()
            cv2.rectangle(ov,(0,0),(FW,FH),(10,10,22),cv2.FILLED)
            cv2.addWeighted(ov,0.78,frame,0.22,0,frame)
            cx = FW//2
            txt(frame,"GESTURE PUZZLE",(cx-210,90),1.9,(0,220,255),3)
            txt(frame,"Hand-tracking drag & drop",(cx-185,125),0.7,(160,160,160),1)
            y=180
            for key,(name,c,r,t) in DIFFICULTIES.items():
                col=(255,220,80) if key=='1' else (255,180,80) if key=='2' else (255,100,80)
                txt(frame,f"[{key}]  {name}  {c}x{r} grid  —  {t}s",(cx-205,y),0.78,col,2)
                best=scores.get(name,[])
                if best:
                    m,s=divmod(int(best[0]),60)
                    txt(frame,f"Best {m}:{s:02}",(cx+145,y),0.55,(80,255,160),1,
                        font=cv2.FONT_HERSHEY_SIMPLEX)
                y+=55
            txt(frame,"Press 1/2/3 to start  |  ESC to quit",
                (cx-225,FH-35),0.65,(140,140,140),1,font=cv2.FONT_HERSHEY_SIMPLEX)
            cv2.imshow("Gesture Puzzle",frame)
            key=cv2.waitKey(1)&0xFF
            if key==27: break
            if chr(key) in DIFFICULTIES: start_game(chr(key))
            continue

        # PLAYING
        if state == 'playing':
            elapsed = time.time()-t_start
            time_left = max(0.0, time_limit-elapsed)

            rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb))

            # Target zone + grid lines
            cv2.rectangle(frame,(TX,TY),(TX+PUZZLE_W,TY+PUZZLE_H),(220,220,220),2)
            for gc in range(1,cols):
                lx=TX+gc*pw; cv2.line(frame,(lx,TY),(lx,TY+PUZZLE_H),(100,100,100),1)
            for gr in range(1,rows):
                ly=TY+gr*ph; cv2.line(frame,(TX,ly),(TX+PUZZLE_W,ly),(100,100,100),1)
            cv2.putText(frame,"DROP ZONE",(TX+4,TY+PUZZLE_H+16),
                        cv2.FONT_HERSHEY_SIMPLEX,0.45,(160,160,160),1)

            # Thumbnail (bottom-right)
            ty2=FH-95; tx2=FW-95
            frame[ty2:ty2+90,tx2:tx2+90]=THUMB
            cv2.rectangle(frame,(tx2,ty2),(tx2+90,ty2+90),(160,160,160),1)
            cv2.putText(frame,"REF",(tx2+28,ty2+90+14),cv2.FONT_HERSHEY_SIMPLEX,0.4,(160,160,160),1)

            # Hand detection
            active = {}   # hand_idx -> (ix,iy,pinching)
            if res.hand_landmarks:
                for hi,lms in enumerate(res.hand_landmarks):
                    for a,b in HAND_CONNECTIONS:
                        p1,p2=lms[a],lms[b]
                        cv2.line(frame,(int(p1.x*FW),int(p1.y*FH)),
                                       (int(p2.x*FW),int(p2.y*FH)),(255,140,0),1)
                    th,idx_l=lms[4],lms[8]
                    tx_=int(th.x*FW);ty_=int(th.y*FH)
                    ix=int(idx_l.x*FW);iy=int(idx_l.y*FH)
                    pinch = math.hypot(ix-tx_,iy-ty_)<35
                    active[hi]=(ix,iy,pinch)
                    dcol=(0,255,80) if pinch else (0,80,255)
                    cv2.circle(frame,(ix,iy),9,dcol,cv2.FILLED)
                    cv2.circle(frame,(ix,iy),9,(255,255,255),1)

                    # Emit trail when dragging
                    if pinch and hi in sel:
                        emit_trail(particles,ix,iy)

                    # Open-palm shuffle detection
                    if is_palm_open(lms,FW,FH) and not pinch:
                        if hi not in palm_timers:
                            palm_timers[hi]=time.time()
                        elif time.time()-palm_timers[hi]>2.0 and shuffle_cd==0:
                            shuffle_pieces()
                            shuffle_cd=60
                            emit_burst(particles,FW//2,FH//2,60)
                    else:
                        palm_timers.pop(hi,None)

            # Palm timers for hands that left frame
            for hi in list(palm_timers):
                if hi not in active: del palm_timers[hi]

            if shuffle_cd>0: shuffle_cd-=1

            # Ghost preview
            for hi,(ix,iy,pinch) in active.items():
                if pinch and hi in sel:
                    hp=sel[hi]
                    gc_=(hp.x-TX+pw//2)//pw; gr_=(hp.y-TY+ph//2)//ph
                    if 0<=gc_<cols and 0<=gr_<rows:
                        gx=TX+int(gc_)*pw; gy=TY+int(gr_)*ph
                        ov2=frame.copy()
                        cv2.rectangle(ov2,(gx,gy),(gx+pw,gy+ph),(0,220,255),cv2.FILLED)
                        cv2.addWeighted(ov2,0.25,frame,0.75,0,frame)
                        cv2.rectangle(frame,(gx,gy),(gx+pw,gy+ph),(0,220,255),2)

            # Drag & drop 
            for hi,(ix,iy,pinch) in active.items():
                if pinch:
                    if hi not in sel:
                        for p in sorted(pieces, key=lambda p: p.is_dragging, reverse=True):
                            if not p.is_placed and p.hovered(ix,iy) and p not in sel.values():
                                sel[hi]=p; p.is_dragging=True; break
                    else:
                        hp=sel[hi]
                        hp.x=ix-pw//2; hp.y=iy-ph//2
                else:
                    if hi in sel:
                        dp=sel.pop(hi); dp.is_dragging=False
                        gc_=int((dp.x-TX+pw//2)//pw); gr_=int((dp.y-TY+ph//2)//ph)
                        if 0<=gc_<cols and 0<=gr_<rows:
                            if (gc_,gr_)==dp.correct_pos:
                                dp.x=TX+gc_*pw; dp.y=TY+gr_*ph
                                dp.is_placed=True; dp.snap_glow=25
                                G['placed_count']+=1; G['streak']+=1
                                emit_burst(particles,TX+gc_*pw+pw//2,TY+gr_*ph+ph//2)
                                if G['placed_count']==rows*cols:
                                    state='win'
                                    G['win_flash']=8
                                    save_score(scores,diff_name,elapsed)
                            else:
                                dp.x,dp.y=dp.home_x,dp.home_y
                                dp.flash=15; G['streak']=0
                                G['penalty_flash']=8

            # Cleanup lost hands
            for hi in [h for h in sel if h not in active]:
                sel[hi].is_dragging=False; del sel[hi]

            # Render pieces
            for p in pieces:
                py1=max(0,int(p.y)); py2=min(FH,int(p.y)+ph)
                px1=max(0,int(p.x)); px2=min(FW,int(p.x)+pw)
                rh,rw=py2-py1,px2-px1
                if rh>0 and rw>0:
                    frame[py1:py2,px1:px2]=p.image[:rh,:rw]
                    if p.is_placed:
                        bc=(0,255,80); bt=2
                    elif p.is_dragging:
                        bc=(0,160,255); bt=3
                    elif p.flash>0:
                        bc=(0,0,255); bt=3
                    else:
                        bc=(0,220,220); bt=1
                    cv2.rectangle(frame,(px1,py1),(px2,py2),bc,bt)
                    if p.snap_glow>0:
                        gv=frame.copy()
                        cv2.rectangle(gv,(px1,py1),(px2,py2),(0,255,100),cv2.FILLED)
                        cv2.addWeighted(gv,p.snap_glow/25*0.4,frame,1-p.snap_glow/25*0.4,0,frame)
                        p.snap_glow-=1
                if p.flash>0: p.flash-=1

            # Particles
            for part in particles[:]:
                part.update(); part.draw(frame)
                if part.life<=0: particles.remove(part)

            #  Penalty screen flash 
            if G['penalty_flash']>0:
                ov3=frame.copy()
                cv2.rectangle(ov3,(0,0),(FW,FH),(0,0,200),cv2.FILLED)
                cv2.addWeighted(ov3,0.25,frame,0.75,0,frame)
                G['penalty_flash']-=1

            #  Shuffle progress indicator 
            for hi,ts in palm_timers.items():
                prog=min(1.0,(time.time()-ts)/2.0)
                if prog>0.05:
                    cv2.rectangle(frame,(FW//2-80,FH-30),(FW//2-80+int(160*prog),FH-15),
                                  (0,200,255),cv2.FILLED)
                    txt(frame,"HOLD TO SHUFFLE",(FW//2-90,FH-35),0.5,(0,200,255),1,
                        font=cv2.FONT_HERSHEY_SIMPLEX)

            # HUD
            cv2.rectangle(frame,(0,0),(FW,52),(15,15,25),cv2.FILLED)
            ratio=time_left/time_limit
            bc=(0,200,80) if ratio>0.5 else (0,160,255) if ratio>0.2 else (0,50,255)
            cv2.rectangle(frame,(0,47),(int(FW*ratio),52),bc,cv2.FILLED)
            m_,s_=divmod(int(time_left),60)
            txt(frame,f"{m_}:{s_:02}",(10,38),1.0,bc,2)
            txt(frame,f"Placed: {G['placed_count']}/{rows*cols}",(FW//2-70,38),0.8,(255,255,255),2,
                font=cv2.FONT_HERSHEY_SIMPLEX)
            if G['streak']>1:
                txt(frame,f"STREAK x{G['streak']}!",(FW-175,38),0.8,(0,220,255),2,
                    font=cv2.FONT_HERSHEY_SIMPLEX)
            txt(frame,diff_name,(FW-165 if G['streak']<=1 else FW-290,38),0.6,(160,160,160),1,
                font=cv2.FONT_HERSHEY_SIMPLEX)

            if time_left<=0: state='timeout'

        # WIN
        if state == 'win':
            if G['win_flash']>0:
                ov=frame.copy(); cv2.rectangle(ov,(0,0),(FW,FH),(255,255,255),cv2.FILLED)
                cv2.addWeighted(ov,G['win_flash']/8*0.7,frame,1-G['win_flash']/8*0.7,0,frame)
                G['win_flash']-=1
            else:
                ov=frame.copy()
                cv2.rectangle(ov,(FW//2-240,FH//2-80),(FW//2+240,FH//2+90),(10,20,10),cv2.FILLED)
                cv2.addWeighted(ov,0.75,frame,0.25,0,frame)
                txt(frame,"PUZZLE SOLVED!",(FW//2-195,FH//2-10),1.8,(0,255,100),3)
                elapsed2=time.time()-t_start; m2,s2=divmod(int(elapsed2),60)
                txt(frame,f"Time: {m2}:{s2:02}",(FW//2-70,FH//2+45),1.0,(255,220,80),2)
                best=scores.get(diff_name,[])
                if best and round(elapsed2,2)==best[0]:
                    txt(frame,"NEW BEST!",(FW//2-75,FH//2+80),0.85,(0,220,255),2,
                        font=cv2.FONT_HERSHEY_SIMPLEX)
                txt(frame,"1/2/3=Play Again  M=Menu  ESC=Quit",
                    (FW//2-230,FH//2+115),0.6,(160,160,160),1,font=cv2.FONT_HERSHEY_SIMPLEX)
                for part in particles[:]:
                    part.update(); part.draw(frame)
                    if part.life<=0: particles.remove(part)
                if len(particles)<80: emit_burst(particles,random.randint(50,FW-50),
                                                  random.randint(50,FH-50),5)

        #  TIMEOUT
        if state == 'timeout':
            ov=frame.copy()
            cv2.rectangle(ov,(0,0),(FW,FH),(20,10,10),cv2.FILLED)
            cv2.addWeighted(ov,0.75,frame,0.25,0,frame)
            txt(frame,"TIME'S UP!",(FW//2-155,FH//2-10),1.8,(0,60,255),3)
            txt(frame,f"Placed {G['placed_count']}/{rows*cols} pieces",
                (FW//2-155,FH//2+45),0.9,(200,200,200),2,font=cv2.FONT_HERSHEY_SIMPLEX)
            txt(frame,"1/2/3=Try Again  M=Menu  ESC=Quit",
                (FW//2-225,FH//2+90),0.6,(140,140,140),1,font=cv2.FONT_HERSHEY_SIMPLEX)

        cv2.imshow("Gesture Puzzle",frame)
        key=cv2.waitKey(1)&0xFF
        if key==27: break
        if key==ord('m'): state='menu'
        if state in ('win','timeout') and chr(key) in DIFFICULTIES:
            start_game(chr(key))

finally:
    cap.release()
    cv2.destroyAllWindows()
    detector.close()