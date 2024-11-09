from music_theory import Chord
import pyaudio,pygame,wave,numpy,sys,datetime
import time as t
from threading import Thread
import homophonic

# Initalization
pygame.init()
wave_file = homophonic.wave_file
framerate = wave_file.getframerate()
music_duration = wave_file.getnframes()/float(framerate)
print("Visualization of this audio.")

# Main Visualization Window
window_size = (1280,720)
font_size = 18
# Functions
def center_distance(size:int):
    return (window_size[0]/2-int(size/2),window_size[1]/2-int(size/2))
def seconds_to_time(seconds):
    return str(datetime.timedelta(seconds=seconds))
# Main Visualization Settings
screen = pygame.display.set_mode(window_size)
pygame.display.set_caption('Visualization')
icon = pygame.image.load('icon\\icon-v.png')
basic_font = pygame.font.SysFont('serife.fon',font_size)

pygame.display.set_icon(icon)
clock = pygame.time.Clock()
keep_going = True
fps = 60

# Initalize the displayer
facesize,bd1bgs,bd1s=60,120,160

face = pygame.Surface((facesize,facesize),flags=pygame.HWSURFACE)
face.fill(color='white')
bass_display1_bg = pygame.Surface((bd1bgs,bd1bgs),flags=pygame.HWSURFACE)
bass_display1 = pygame.Surface((bd1s,bd1s),flags=pygame.HWSURFACE)

frame = 0
beat = 0
is_display_beat = False
music_duration_text = seconds_to_time(music_duration)

# Play music
pygame.mixer.init()
info4_text = "Now playing: "+homophonic.filename
music = pygame.mixer.Sound(homophonic.filename)
pygame.mixer.music.load(homophonic.filename.replace('.wav','.mp3'))
pygame.mixer.music.play(0,0)
st = t.time()

# Main Loop
while keep_going:
    # Texts
    info_text = "DEBUG--Time: "
    info2_text = "Real Time: "
    info3_text = "Delay: "
    # Time and FPS Settings, Refill screen
    clock.tick(fps)
    screen.fill('black')
    time = frame/fps
    # Playing Progress
    text_time = seconds_to_time(time)
    real_time = t.time() - st
    deltatime = int((time-real_time)*1e4)/1e1

    # Beat Settings
    nextbeat = homophonic.beat_times[beat]

    lastbeat = 0 if beat==0 else homophonic.beat_times[beat-1]
    deltabeat = nextbeat-lastbeat
    till_nextbeat = nextbeat-time
    this_beat_progress=bp=till_nextbeat/deltabeat
    info_text+=text_time+' / '+music_duration_text

    # EventHandler
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            keep_going = False

    # Control Beats Displays
    if is_display_beat:
        is_display_beat=False

    if till_nextbeat <= 0:
        is_display_beat = True
        pygame.mixer.music.pause()
        pygame.mixer.music.play(0,time)
        print("Send beat:",beat)
        beat+=1

    # Renderers
    info2_text += seconds_to_time(real_time)
    info3_text += str(deltatime)+'ms'
    textObj = basic_font.render(info_text,
                                True, (255, 255, 255), (0, 0, 0))
    textObj2 = basic_font.render(info2_text,
                                True, (255, 255, 255), (0, 0, 0))
    textObj3 = basic_font.render(info3_text,
                                True, (255, 255, 255), (0, 0, 0))
    textObj4 = basic_font.render(info4_text,
                                 True, (255, 255, 255), (0, 0, 0))

    c = abs((bp) * 255)


    # Beat Displays Settings
    face.fill(color=(c,c,c))
    bass_display1_bg.fill(color=(0,0,0))
    bass_display1.fill(color=(0,0,0))
    if bp > 0.5 and beat/2==int(beat/2):
        bass_display1.fill(color=(255,255,255))

    # Blit patterns
    screen.blit(textObj,(0,0))
    screen.blit(textObj2,(0,font_size))
    screen.blit(textObj3,(0,2*font_size))
    screen.blit(textObj4, (0,3*font_size))

    # Center rectangles
    screen.blit(bass_display1,center_distance(bd1s))
    screen.blit(bass_display1_bg,center_distance(bd1bgs))
    # Tempo display
    screen.blit(face, center_distance(facesize))

    pygame.display.flip()

    # Detect delay
    if deltatime >= 0.01:
        pygame.time.wait(round(deltatime))

    frame+=1

    # When end:
    if beat == len(homophonic.beat_times)-1:
        keep_going=False

# Exit
pygame.quit()
sys.exit()
