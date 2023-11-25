[
  .data[] 
  | .cards[] 
  | select((has("side") == false or .side == "a") and .legalities.modern == "Legal" and .language == "English" and has("name") == true) 
  | del(.sourceProducts, .setCode, .purchaseUrls, .foreignData, .artist,.artistIds,.availability,.boosterTypes,.colorIdentity,.colors,.convertedManaCost,.edhrecRank,.edhrecSaltiness,.finishes,.frameVersion,.hasFoil,.hasNonFoil,.identifiers,.isReprint,.layout,.legalities,.manaValue,.number,.originalText,.originalType,.printings,.type,.uuid,.variations,.flavorText,.borderColor,.language,.rulings,.asciiName,.cardParts,.colorIndicator,.duelDeck,.faceConvertedManaCost,.faceFlavorName,.faceManaValue,.faceName,.flavorName,.frameEffects,.hasAlternativeDeckLimit,.isFullArt,.isOnlineOnly,.isPromo,.isStarter,.isStorySpotlight,.isTimeshifted,.leadershipSkills,.originalReleaseDate,.otherFaceIds,.promoTypes,.rebalancedPrintings,.securityStamp,.side,.subsets,.watermark)
  ]
| unique_by(.name)

