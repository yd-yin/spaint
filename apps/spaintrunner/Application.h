/**
 * spaintrunner: Application.h
 */

#ifndef H_SPAINTRUNNER_APPLICATION
#define H_SPAINTRUNNER_APPLICATION

#include <SDL.h>

#include <spaint/main/SpaintEngine.h>
#include <spaint/util/SharedPtr.h>

/**
 * \brief The main application class for spaintrunner.
 */
class Application
{
  //#################### TYPEDEFS ####################
private:
  typedef spaint::shared_ptr<void> SDL_GLContext_Ptr;
  typedef spaint::shared_ptr<SDL_Window> SDL_Window_Ptr;

  //#################### PRIVATE VARIABLES ####################
private:
  /** The OpenGL context for the main window. */
  SDL_GLContext_Ptr m_context;

  /** The main window for the application. */
  SDL_Window_Ptr m_window;

  //#################### CONSTRUCTORS ####################
public:
  /**
   * \brief Constructs the application.
   *
   * \param spaintEngine The spaint engine that the application should use.
   */
  explicit Application(const spaint::SpaintEngine_Ptr& spaintEngine);

  //#################### PUBLIC MEMBER FUNCTIONS ####################
public:
  /**
   * \brief Runs the application.
   */
  void run();

  //#################### PRIVATE MEMBER FUNCTIONS ####################
private:
  /**
   * \brief Processes any SDL events (e.g. those generated by user input).
   *
   * \return true, if the application should continue running, or false otherwise.
   */
  bool process_events();

  /**
   * \brief Renders the scene into the main window.
   */
  void render() const;

  /**
   * \brief Sets up the viewport and projection matrix.
   */
  void setup();
};

#endif
