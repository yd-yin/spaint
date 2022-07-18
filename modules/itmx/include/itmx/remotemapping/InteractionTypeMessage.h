/**
 * itmx: InteractionTypeMessage.h
 * Copyright (c) Torr Vision Group, University of Oxford, 2018. All rights reserved.
 */

#ifndef H_ITMX_INTERACTIONTYPEMESSAGE
#define H_ITMX_INTERACTIONTYPEMESSAGE

#include <tvgutil/net/SimpleMessage.h>

namespace itmx {

//#################### ENUMERATIONS ####################

/**
 * \brief The values of this enumeration denote the different types of interaction that a mapping client can have with a mapping server.
 *
 * \note  Do not change the enum values without making sure that all clients have been updated accordingly.
 */
enum InteractionType
{
  /** An interaction in which the client asks the server to send across its rendered image of the scene for that client. */
  IT_GETRENDEREDIMAGE = 0,

  /** An interaction in which the client asks the server whether it has ever rendered an image of the scene for that client. */
  IT_HASRENDEREDIMAGE = 1,

  /** An interaction in which the client sends a single RGB-D frame to the server. */
  IT_SENDFRAME = 2,

  /** An interaction in which the client sends a new rendering request to the server. */
  IT_UPDATERENDERINGREQUEST = 3,
};

//#################### TYPES ####################

/**
 * \brief An instance of this type represents a message containing the way in which a mapping client next wants to interact with a mapping server.
 */
typedef tvgutil::SimpleMessage<InteractionType> InteractionTypeMessage;

}

#endif
