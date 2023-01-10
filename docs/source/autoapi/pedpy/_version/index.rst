:orphan:

:py:mod:`pedpy._version`
========================

.. py:module:: pedpy._version

.. autoapi-nested-parse::

   Git implementation of _version.py.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pedpy._version.VersioneerConfig



Functions
~~~~~~~~~

.. autoapisummary::

   pedpy._version.get_keywords
   pedpy._version.get_config
   pedpy._version.register_vcs_handler
   pedpy._version.run_command
   pedpy._version.versions_from_parentdir
   pedpy._version.git_get_keywords
   pedpy._version.git_versions_from_keywords
   pedpy._version.git_pieces_from_vcs
   pedpy._version.plus_or_dot
   pedpy._version.render_pep440
   pedpy._version.render_pep440_branch
   pedpy._version.pep440_split_post
   pedpy._version.render_pep440_pre
   pedpy._version.render_pep440_post
   pedpy._version.render_pep440_post_branch
   pedpy._version.render_pep440_old
   pedpy._version.render_git_describe
   pedpy._version.render_git_describe_long
   pedpy._version.render
   pedpy._version.get_versions



Attributes
~~~~~~~~~~

.. autoapisummary::

   pedpy._version.LONG_VERSION_PY
   pedpy._version.HANDLERS


.. py:function:: get_keywords()

   Get the keywords needed to look up the version information.


.. py:class:: VersioneerConfig

   Container for Versioneer configuration parameters.


.. py:function:: get_config()

   Create, populate and return the VersioneerConfig() object.


.. py:exception:: NotThisMethod

   Bases: :py:obj:`Exception`

   Exception raised if a method is not valid for the current scenario.


.. py:data:: LONG_VERSION_PY
   :annotation: :Dict[str, str]

   

.. py:data:: HANDLERS
   :annotation: :Dict[str, Dict[str, Callable]]

   

.. py:function:: register_vcs_handler(vcs, method)

   Create decorator to mark a method as the handler of a VCS.


.. py:function:: run_command(commands, args, cwd=None, verbose=False, hide_stderr=False, env=None)

   Call the given command(s).


.. py:function:: versions_from_parentdir(parentdir_prefix, root, verbose)

   Try to determine the version from the parent directory name.

   Source tarballs conventionally unpack into a directory that includes both
   the project name and a version string. We will also support searching up
   two directory levels for an appropriately named parent directory


.. py:function:: git_get_keywords(versionfile_abs)

   Extract version information from the given file.


.. py:function:: git_versions_from_keywords(keywords, tag_prefix, verbose)

   Get version information from git keywords.


.. py:function:: git_pieces_from_vcs(tag_prefix, root, verbose, runner=run_command)

   Get version from 'git describe' in the root of the source tree.

   This only gets called if the git-archive 'subst' keywords were *not*
   expanded, and _version.py hasn't already been rewritten with a short
   version string, meaning we're inside a checked out source tree.


.. py:function:: plus_or_dot(pieces)

   Return a + if we don't already have one, else return a .


.. py:function:: render_pep440(pieces)

   Build up version string, with post-release "local version identifier".

   Our goal: TAG[+DISTANCE.gHEX[.dirty]] . Note that if you
   get a tagged build and then dirty it, you'll get TAG+0.gHEX.dirty

   Exceptions:
   1: no tags. git_describe was just HEX. 0+untagged.DISTANCE.gHEX[.dirty]


.. py:function:: render_pep440_branch(pieces)

   TAG[[.dev0]+DISTANCE.gHEX[.dirty]] .

   The ".dev0" means not master branch. Note that .dev0 sorts backwards
   (a feature branch will appear "older" than the master branch).

   Exceptions:
   1: no tags. 0[.dev0]+untagged.DISTANCE.gHEX[.dirty]


.. py:function:: pep440_split_post(ver)

   Split pep440 version string at the post-release segment.

   Returns the release segments before the post-release and the
   post-release version number (or -1 if no post-release segment is present).


.. py:function:: render_pep440_pre(pieces)

   TAG[.postN.devDISTANCE] -- No -dirty.

   Exceptions:
   1: no tags. 0.post0.devDISTANCE


.. py:function:: render_pep440_post(pieces)

   TAG[.postDISTANCE[.dev0]+gHEX] .

   The ".dev0" means dirty. Note that .dev0 sorts backwards
   (a dirty tree will appear "older" than the corresponding clean one),
   but you shouldn't be releasing software with -dirty anyways.

   Exceptions:
   1: no tags. 0.postDISTANCE[.dev0]


.. py:function:: render_pep440_post_branch(pieces)

   TAG[.postDISTANCE[.dev0]+gHEX[.dirty]] .

   The ".dev0" means not master branch.

   Exceptions:
   1: no tags. 0.postDISTANCE[.dev0]+gHEX[.dirty]


.. py:function:: render_pep440_old(pieces)

   TAG[.postDISTANCE[.dev0]] .

   The ".dev0" means dirty.

   Exceptions:
   1: no tags. 0.postDISTANCE[.dev0]


.. py:function:: render_git_describe(pieces)

   TAG[-DISTANCE-gHEX][-dirty].

   Like 'git describe --tags --dirty --always'.

   Exceptions:
   1: no tags. HEX[-dirty]  (note: no 'g' prefix)


.. py:function:: render_git_describe_long(pieces)

   TAG-DISTANCE-gHEX[-dirty].

   Like 'git describe --tags --dirty --always -long'.
   The distance/hash is unconditional.

   Exceptions:
   1: no tags. HEX[-dirty]  (note: no 'g' prefix)


.. py:function:: render(pieces, style)

   Render the given version pieces into the requested style.


.. py:function:: get_versions()

   Get version information or return default if unable to do so.


