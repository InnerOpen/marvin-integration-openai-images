# OpenAI Images

Makes pictures from a prompt, or changes a picture you have, through OpenAI's image API. Every generation and edit
is a paid call, so both actions need a person's approval: in a workflow they run right after an approved **Approval**
step; from the bubble or Ask they wait for the user's yes.

## Connect it

- **API key**: an OpenAI key (`sk-…`) whose project has image access.
- **Model**: `gpt-image-2` for finals (the current model), `gpt-image-1-mini` for drafts and variants (under half the
  price). There is no "gpt-image-2 mini". `dall-e-*` still work but take none of the options below.
- **Size**: `1024x1024` (square), `1536x1024` (landscape), `1024x1536` (portrait) or `auto`.
- **Quality** (`auto`, `low`, `medium`, `high`) and **Picture format** (`png`, `jpeg`, `webp`) are the connection's
  defaults; a step may override them per call.
- **Check connection** really checks: the key is tried and the model must be one the key can use.

## Cost

Billed by OpenAI per picture, mostly by output tokens: on `gpt-image-1-mini` a 1024×1024 picture is about 4,000
output tokens (a few cents); `gpt-image-2` is roughly two to five times that, more at `high` quality and larger
sizes. The usage comes back on every call, so Marvin's AI usage panel and the monthly limit see it. `low` quality is
the cheap way to try a prompt; `high` is for the one you keep.

## Actions

- **Generate image** (`generate`): `prompt` (required), `count` (default 1), and optionally `quality`,
  `output_format`, `background` (`transparent` needs png or webp). Returns `images: [{image_b64}]` plus `usage`.
  In a workflow Marvin keeps each picture as an asset (prompt, model and run recorded) and the step's output carries
  the asset's id and url; `attach_as` links it to the run's entry in that role.
- **Edit an image** (`edit`): `prompt`, the picture (filled in from the step's `source`), `background`
  (`auto` / `transparent` / `opaque`), `fidelity` (`high` keeps the details). Removes a background, sets the thing in
  a new scene, makes a seasonal variant. Returns one `image_b64`, kept as a version of the source.
- **List image models** (`list_models`): free. The image models this key can use, the connection's model, and
  whether it is one of them — the plain way to test the connection.

## In workflows

- Always an **Approval** step first; the step after it runs only once approved.
- A picture for Instagram: `output_format: jpeg` on the generate step saves a convert step (the feed takes JPEG only).
- Drafts cheaply: a connection on `gpt-image-1-mini` named for it, and the recipe's `image_integration` variable
  pointing at it; finals on the `gpt-image-2` connection.
- Prompts: say what the picture is for and the style; the entry's title and summary are the usual ingredients
  (`${entry.title}`, `${entry.summary}`).

## What it can't do

- Upscale, or make a video.
- Generate without approval, or outside the workspace's AI budget.
- Return a picture larger than the sizes above; resize or crop with Marvin's `media` step afterwards.
