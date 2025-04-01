from pathlib import Path
import numpy as np

import dotenv

from strategies.chunks_from_text import ChunkFromTextStrategy
from core_types import PartialChunk, PartialDocument, Task
from storage import temporary_db

dotenv.load_dotenv()


doc = """
So what you'll see on the screen here is what I call the deep life stack. Right now I have it empty. We're gonna fill in the details as we go along. The stack has four different levels to it. I'll highlight those. Got level one, two, three, four. The way I'm conceptualizing in DeepLife stack is sequential. You start with developing the bottom layer of the stack, then you move up to the second layer, then the third, then the fourth, and then we're gonna iterate, and we'll get into that soon. Alright. So what happens on the first layer of the DeepLife stack? This is the first big change or breakthrough I would say I've had when thinking about the DeepLife more recently, What I'm gonna put at this bottom layer is gonna be discipline. That's gonna mean two things. But let me let me say what my goal is here. I'm realizing when it comes to cultivating a different type of life, any type of transformation, you have to first change your self identification to be the type of person who is able to persist with things that are difficult in the moment in pursuit of a greater good down the line. And I think it's very easy for people like me who give advice for a living, who've been doing this for a long time, to take for granted that that's what we do already. But this is actually for most people, Maybe the most critical step is transitioning from someone who says, look, this is not me. I don't have discipline. I'm not really able to pursue goals unless I feel really excited about it in the moment. How do we shift that self identity? And as long time listeners of the show know, I really do see discipline as an identity. It is not something you do. It is an identity. You see yourself as someone who is disciplined or you don't that require some cultivation. So at the very bottom of the deep life stack, and this is why I've highlighted this, you would get started by putting some elements into your life that required discipline to accomplish. And, and it, it doesn't really matter when we're first beginning here, What these are, you just wanna push them to be past what's trivial, but still south of intractable. So where you're starting from might depend where, how, ambitious these initial bits of discipline are. So this is where you might say, look. I'm gonna train for a five k. I am going to read five books a month. You're you're you're trying to find something that's gonna require some discipline. I'm gonna overhaul my nutrition. I'm gonna do something new. I'm going to do this workout routine, try to hit a streak on Peloton, whatever it is. You're calibrating it to where you are. And I don't really care at first, the content of these things you're pursuing with discipline, this is identity formation. And so that's where we get started. You take a couple of things in your life. You say, how can I make progress on this every day? And if it's too hard, you find something easier until you can move up to something harder, but you're establishing discipline. The second piece here is you're going to establish your route for everything we're about to do, a directory, a folder, a drawer, and a desk where it's gonna be the one place where you keep track of everything that you've committed to do in your life, your rules, your systems, your goals, so that you're going to initialize this route to your ultimate life planning processes with these initial discipline projects. So at the beginning, you could just have a folder on your desktop. You could have a drawer where you're just writing down, here's my disciplines. I'm working on these two things. Here's what I do every day towards them. Them. This is going to grow as we move to the deep life stack, but you're establishing here in the discipline step, here's where I keep track of what I commit to, and you're starting to practice having commitments that are about long term value, not what you wanna do in the short term. So already we're a little bit different than standard thinking about lifestyle designs because we're not starting with the decisions. We're not starting with the let's quit my job. I wanna move to the country. We're recognizing that there is some effacement that has to happen first. There's some preparation that has to happen first. We don't wanna jump into the decisions till we develop the the self first. So that's the first layer of the stack.
"""


async def test_chunks_from_text_strategy(tmp_path: Path):
    strategy = ChunkFromTextStrategy()
    assert strategy.NAME == "chunk_from_text"

    doc_path = tmp_path / "doc.txt"
    doc_path.write_text(doc)
    doc_id = 1

    task = Task(
        id=-1,
        strategy=strategy.NAME,
        document_id=doc_id,
        args=str(doc_path),
    )

    with temporary_db() as db:
        await strategy.process_all([task])

        chunks = db.get_chunks_by_document_id(doc_id)
        assert len(chunks) > 0

        chunk_lengths = [len(chunk.content) for chunk in chunks]
        assert all(length > 0 for length in chunk_lengths)
        assert all(length < 2000 for length in chunk_lengths)

        text_from_chunks = "\n".join(chunk.content for chunk in chunks)

        # Length should be +- 10% of the original text
        assert abs(len(text_from_chunks) - len(doc)) / len(doc) < 0.1, f"Text length mismatch: {len(text_from_chunks)} vs {len(doc)}"
