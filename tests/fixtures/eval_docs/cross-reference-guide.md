# Cross-Reference Guide

## Deployment-to-Service Mapping

Project Lantern is the deployment pipeline for the checkout service. Its rollback procedures are executed by the checkout team owner listed in the service ownership roster. The one-percent error rate threshold applies to every critical-tier service on the roster during a Lantern release.

## Shared Infrastructure

The home lab PostgreSQL server that runs weekly base backups also hosts the analytics database. The analytics service is owned by Jon, who coordinates through the team data channel and reviews on Wednesdays. A recovery point objective of fifteen minutes covers both the personal database and the analytics service.

## Schedule Coordination

Friday afternoon review sessions are used to plan the following week of travel. The preferred morning travel window aligns with the deep work period before lunch, and itinerary changes are batched with administrative messages after noon.
