from pydantic import BaseModel

class DesktopItem(BaseModel):
    id: str | None = None
    name: str
    attributes: str | None = None
    operated: bool | None = False


STUDY_DESK = [
    DesktopItem(id='lamp-1', name='lamp', attributes='white, phone holder style', operated=False),
    DesktopItem(id='mouthwash-1', name='mouthwash', attributes='transparent bottle with blue cap', operated=False),
    DesktopItem(id='toolset-1', name='toolset', attributes='orange and transparent top', operated=False),
    DesktopItem(id='yogurt-1', name='yogurt', attributes='white, ceramic bottle', operated=False),
    DesktopItem(id='baseball-1', name='baseball', attributes='white with red stitching, MIT logo', operated=False),
    DesktopItem(id='doll-1', name='doll', attributes='gray, owl-shaped doll', operated=False),
    DesktopItem(id='tissues-1', name='tissues', attributes='blue and white package', operated=False),
    DesktopItem(id='thermos-1', name='thermos', attributes='light green', operated=False),
    DesktopItem(id='gamecard-1', name='gamecard', attributes='green Xbox game case', operated=False),
    DesktopItem(id='umbrella-1', name='umbrella', attributes='black and white striped', operated=False)
]


OFFICE_DESK = [
    DesktopItem(id='bottle-1', name='bottle', attributes='transparent, plastic, with green cap', operated=False),
    DesktopItem(id='notebook-1', name='notebook', attributes='cork cover', operated=False),
    DesktopItem(id='book-1', name='book', attributes='white, pink and green cover, "IDEO"', operated=False),
    DesktopItem(id='book-2', name='book', attributes='white cover, "Design as Art"', operated=False),
    DesktopItem(id='fan-1', name='fan', attributes='small, white, round', operated=False),
    DesktopItem(id='measure-1', name='measure', attributes='retractable tape measure', operated=False),
    DesktopItem(id='tape-1', name='tape', attributes='roll of black electrical tape', operated=False),
    DesktopItem(id='remote-1', name='remote', attributes='white, air conditioner remote', operated=False),
    DesktopItem(id='sunglasses-1', name='sunglasses', attributes='brown lenses, black frame', operated=False),
    DesktopItem(id='keyboard-1', name='keyboard', attributes='black, Logitech brand', operated=False),
    DesktopItem(id='basket-1', name='basket', attributes='silver, wire mesh', operated=False),
    DesktopItem(id='bin-1', name='bin', attributes='blue, plastic', operated=False)
]

BAR_COUNTER = [
    DesktopItem(id='plate-1', name='plate', attributes='black, with white bird pattern', operated=False),
    DesktopItem(id='bowl-1', name='bowl', attributes='off-white, ceramic', operated=False),
    DesktopItem(id='bowl-2', name='bowl', attributes='small, white, with gold rim', operated=False),
    DesktopItem(id='brush-1', name='brush', attributes='gray sponge brush on a stick', operated=False),
    DesktopItem(id='sauce-1', name='sauce', attributes='glass jar with red cap', operated=False),
    DesktopItem(id='noodles-1', name='noodles', attributes='instant noodle cup, red and black', operated=False),
    DesktopItem(id='detergent-1', name='detergent', attributes='bottle of yellow dish soup', operated=False),
    DesktopItem(id='oil-1', name='oil', attributes='glass bottle of cooking oil', operated=False),
    DesktopItem(id='spoon-1', name='spoon', attributes='white, ceramic', operated=False),
    DesktopItem(id='sponge-1', name='sponge', attributes='yellow, wavy shape', operated=False),
    DesktopItem(id='sponge-2', name='sponge', attributes='yellow, wavy shape', operated=False),
    DesktopItem(id='lemon-1', name='lemon', attributes='yellow, whole fruit', operated=False),
]