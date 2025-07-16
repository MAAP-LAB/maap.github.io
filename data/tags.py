tag_detail = {
  "genre": [
    "pucnk"
    "soundtrack",
    "electronic",
    "ambient",
    "rnb",
    "dance",
    "lounge",
    "house",
    "trance",
    "progressive",
    "classical",
    "rap",
    "techno",
    "indie",
    "rock",
    "pop",
    "jazz",
    "hiphop",
    "metal",
    "blues",
    "reggae",
    "country",
    "folk",
  ],


  "mood/theme": [
    "dream",
    "emotional",
    "film",
    "energetic",
    "inspiring",
    "love",
    "melancholic",
    "relaxing",
    "sad",
    "romantic",
    "hopeful",
    "motivational",
    "happy",
    "sport",
    "children",
    "trailer",
    "joyful",
    "christmas",
    "epic",
    "motivational",
    "dark",
    "scifi",
    "festive"
  ],
  "instrument": [
    "synthesizer",
    "drums",
    "strings",
    "guitar",
    "piano",
    "saxophone",
    "beat",
    "violin",
    "bell",
    "percussion",
    "choir",
    "pad",
    "flute",
    "electricguitar",
    "keyboard",
    "horn",
    "guitar",
    "bongo",
    "accordion",
    "bass",
    "clavier"
  ]
}

if __name__ == "__main__":
    for tag, genres in tag_detail.items():
        print(f"{tag}: {', '.join(genres)}")
    # Example usage
    # print(tag_detail["genre"])
    # print(tag_detail["mood/theme"])
    # print(tag_detail["instrument"])