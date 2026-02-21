"""All LLM prompts as string constants. Never embed prompts elsewhere."""

SCENE_DESCRIBE_PROMPT = """You are an expert construction analyst with deep knowledge of building phases and spatial geometry.

Analyze this construction site image. You may also have a depth map showing the 3D geometry.

Return a JSON object with this exact structure:
{
  "construction_phase": "framing|rough-in|drywall|finishing|complete",
  "room_dimensions": {
    "estimated_width_ft": <number>,
    "estimated_depth_ft": <number>,
    "ceiling_height_ft": <number>
  },
  "elements_present": [
    {
      "type": "stud_wall|ceiling_joist|floor_joist|concrete_slab|subfloor|electrical_box|pipe|hvac_duct|window_rough_opening|door_rough_opening|temporary_bracing",
      "location": "north_wall|south_wall|east_wall|west_wall|ceiling|floor|center",
      "description": "<brief description of what you see>"
    }
  ],
  "surfaces": {
    "walls": "bare_studs|osb|drywall_unfinished|drywall_painted|concrete_block|brick",
    "floor": "concrete_slab|subfloor_plywood|hardwood|tile|carpet",
    "ceiling": "exposed_joists|drywall_unfinished|drywall_painted|drop_ceiling"
  },
  "lighting_conditions": "good|poor|mixed",
  "camera_angle": "wide_angle|close_up|corner_shot|looking_up|looking_down",
  "notable_observations": "<any other relevant construction details, measurements, or spatial observations>"
}

Be specific about quantities (e.g., "3 electrical boxes visible on north wall") and positions.
Return ONLY the JSON, no other text."""


FUTURE_STATE_PROMPT = """You are an expert interior designer and construction specialist.

Given this construction scene analysis:
{scene_json}

Describe the FINISHED state this space will reach when construction is complete.
Apply standard construction practices and building codes.

Return a JSON object with this exact structure:
{
  "finished_description": "<one paragraph describing the completed room>",
  "layers": {
    "walls": {
      "material": "drywall_painted|plaster|tile|wood_paneling",
      "color": "<color description>",
      "texture": "smooth|orange_peel|knockdown",
      "details": ["<list of wall details like outlets, switches, trim>"]
    },
    "floor": {
      "material": "hardwood|tile|carpet|concrete_polished|vinyl_plank",
      "color": "<color description>",
      "pattern": "<if applicable>"
    },
    "ceiling": {
      "material": "drywall_painted|drop_ceiling|exposed_concrete",
      "color": "<color>",
      "fixtures": ["<list of ceiling fixtures: lights, vents, sprinklers>"]
    },
    "electrical": [
      {
        "type": "outlet|switch|light_fixture|panel",
        "location": "<wall and position description>",
        "height_inches_from_floor": <number>,
        "quantity": <number>
      }
    ],
    "fixtures": [
      {
        "type": "door|window|baseboard|crown_molding|hvac_vent",
        "location": "<description>",
        "dimensions": "<if relevant>"
      }
    ]
  },
  "inpaint_prompts": {
    "walls": "<detailed SD prompt for inpainting the walls>",
    "floor": "<detailed SD prompt for inpainting the floor>",
    "ceiling": "<detailed SD prompt for inpainting the ceiling>",
    "electrical": "<detailed SD prompt for electrical elements>"
  }
}

For inpaint_prompts, be highly specific: e.g. 'smooth white drywall with eggshell latex paint, baseboard trim, single electrical outlet at 12 inches, photorealistic, 8k, construction photography' rather than generic descriptions.

Return ONLY the JSON, no other text."""


INPAINT_WALL_PROMPT = (
    "smooth white drywall wall with eggshell latex paint finish, "
    "baseboard molding at floor, photorealistic interior construction photography, "
    "8k resolution, sharp focus, professional lighting"
)

INPAINT_FLOOR_PROMPT = (
    "engineered hardwood flooring, light oak color, smooth finish, "
    "photorealistic interior, 8k, sharp focus, professional photography"
)

INPAINT_CEILING_PROMPT = (
    "smooth white painted drywall ceiling, recessed LED lighting, "
    "photorealistic interior, 8k, sharp focus, professional photography"
)

INPAINT_ELECTRICAL_PROMPT = (
    "standard duplex electrical outlet cover plate, white, mounted at 12 inches from floor, "
    "photorealistic, sharp focus, 8k"
)

NEGATIVE_PROMPT = (
    "blurry, distorted, low quality, unrealistic, cartoon, sketch, "
    "construction materials visible, studs, framing, raw wood, concrete"
)
