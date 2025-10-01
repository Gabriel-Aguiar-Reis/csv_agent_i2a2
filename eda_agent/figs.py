import io
import base64
# Função utilitária para converter figura matplotlib para base64
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_bytes = buf.read()
    buf.close()
    return base64.b64encode(img_bytes).decode("utf-8")