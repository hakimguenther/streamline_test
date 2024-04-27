import streamlit as st
import pygame
import pygame.gfxdraw
from PIL import Image


st.set_page_config(
    layout="wide",
    page_title="Pygame-CE Drawing Sandbox",
    page_icon="🎨",
)



st.title("Pygame-CE Drawing Sandbox 🎨:snake:")
st.write("Experiment and test out Pygame (Community Edition) static drawing functions with a real-time preview.")
st.markdown("In this sandboxed environment, `pygame`, `pygame.gfxdraw`, and `numpy` are imported and a limited set of Python `builtins` are available for use.")
st.markdown("The display surface variable is named `SCREEN`, it's size is `(800, 600)` and is accessible via `SCREEN_SIZE`, and uploaded image surface accessible via `IMAGE_SURFACE`.")
st.markdown("Feel free to tag or message me on the [Pygame-CE Discord](https://discord.com/channels/772505616680878080/772505616680878083) `@Djo` for any questions or suggestions. The source code for this Streamlit app is available on GitHub by following the icon link at the top right of the page.")



status = st.container()
col1, col2 = st.columns(2)

pygame.init()
pygame.font.init()

SCREEN_SIZE = (800, 600)
SCREEN = pygame.Surface(SCREEN_SIZE, pygame.SRCALPHA)
SCREEN.fill((0, 0, 0))


try:
    exec("st.title(\"Pygame-CE Drawing Sandbox 🎨:snake:\")")
    status.success('Code executed successfully')
except Exception as e:
    status.error(f"{type(e).__name__}: {e}")

with col2:
    img = Image.frombytes('RGB', SCREEN_SIZE, pygame.image.tobytes(SCREEN, 'RGB'))
    st.image(img)